import json
import math
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation import (
    ACTION_REJECT_PROVISIONAL,
    ACTION_RETAIN,
    CoverageCandidateReconciliationConfig,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_report import (
    POLICY_MODE_EVIDENCE_ONLY,
    build_coverage_candidate_reconciliation_report,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_reconciliation_application import (
    POLICY_MODE_BOUNDED_NEGATIVE_VISIBILITY_REGISTRY_REJECTION,
    apply_negative_visibility_reconciliation_report,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_lifecycle import (
    evaluate_exact_two_lidar_checkpoint,
)
from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    build_exact_two_camera_candidate_snapshot,
    evaluate_exact_two_camera_admission,
    exact_two_camera_admission_sha256,
    new_exact_two_camera_handoff,
    validate_live_registry_binding,
)
from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import (
    write_content_hashed_json,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json,
)
from scripts.aufgabe04.navigation.coverage.coverage_stop_perception_admission import (
    prepare_coverage_visibility_reconciliation,
)
from scripts.aufgabe04.navigation.coverage.coverage_visibility_reporting import (
    CoverageVisibilityEvidence,
)
from scripts.aufgabe04.navigation.planning.map_io import MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    REJECTION_BASIS_NEGATIVE_VISIBILITY,
    STATUS_PENDING_CAMERA,
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
    STATUS_PROVISIONAL,
    STATUS_REJECTED,
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    SurveyViewpoint,
    mark_viewpoint_visited,
    load_coverage_survey_plan,
    load_survey_progress,
    load_stand_survey_registry,
    new_survey_progress,
    stand_survey_registry_sha256,
    write_coverage_survey_plan,
    write_stand_survey_registry,
    write_survey_progress,
)
from scripts.aufgabe04.perception.lidar_visibility_evidence import (
    lidar_visibility_receipt_from_scan,
    visibility_receipts_sha256,
)
from scripts.aufgabe04.real_robot.mission.session_manifest import (
    COVERAGE_SURVEY_TERMINAL_CHECKPOINT,
    admit_autonomous_session_manifest,
    publish_coverage_checkpoint,
)


MAP_SHA256 = "a" * 64
CONFIG_SHA256 = "b" * 64
SURVEY_ID = "survey_01"
SOURCE_VIEWPOINT = "viewpoint_01"
CHECK_VIEWPOINT = "viewpoint_02"
SCAN_POSE = Pose2D(0.05, 0.05, 0.0)
CANDIDATE_POSE = Pose2D(0.85, 0.05, 0.0)
ANGLE_MIN = -math.pi
ANGLE_INCREMENT = math.radians(1.0)
SCAN_COUNT = 361
TARGET_INDEX = 180


def _grid() -> OccupancyGrid:
    rows = tuple(tuple(0 for _x in range(30)) for _y in range(30))
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.10,
            origin=(-1.0, -1.0, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.196,
            mode="trinary",
        ),
        width=30,
        height=30,
        cells=rows,
    )


def _candidate_cell() -> GridCell:
    return GridCell(18, 10)


def _plan(*, expected_count: int = 5) -> CoverageSurveyPlan:
    candidate_cell = _candidate_cell()
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        arena_bounds=ArenaBounds(),
        config=CoverageSurveyConfig(
            lane_count=1,
            expected_stand_count=expected_count,
            exact_inspection_point_count=2,
        ),
        viewpoints=(
            SurveyViewpoint(
                viewpoint_id=SOURCE_VIEWPOINT,
                pose=Pose2D(0.05, -0.45, math.pi / 2.0),
                cell=GridCell(10, 5),
                visible_cells=(candidate_cell,),
            ),
            SurveyViewpoint(
                viewpoint_id=CHECK_VIEWPOINT,
                pose=SCAN_POSE,
                cell=GridCell(10, 10),
                visible_cells=(candidate_cell,),
            ),
        ),
        surveyable_cells=(candidate_cell,),
        planned_covered_cells=(candidate_cell,),
        planned_coverage_ratio=1.0,
    )


def _candidate(
    candidate_uid: str = "survey_candidate_0001",
    *,
    viewpoint_ids: tuple[str, ...] = (SOURCE_VIEWPOINT,),
    status: str = STATUS_PROVISIONAL,
) -> SurveyCandidate:
    suffix = candidate_uid.rsplit("_", 1)[-1]
    return SurveyCandidate(
        candidate_uid=candidate_uid,
        x_m=CANDIDATE_POSE.x_m,
        y_m=CANDIDATE_POSE.y_m,
        radius_m=0.06,
        uncertainty_m=0.02,
        keepout_radius_m=0.31,
        confidence=0.91,
        hit_count=16,
        first_seen_sec=1.0,
        last_seen_sec=2.0,
        source_observation_ids=(f"observation_{suffix}",),
        viewpoint_ids=viewpoint_ids,
        status=status,
    )


def _registry(
    candidates: tuple[SurveyCandidate, ...] | None = None,
) -> StandSurveyRegistry:
    return StandSurveyRegistry(
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
        survey_id=SURVEY_ID,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        candidates=candidates or (_candidate(),),
    )


def _complete_progress(plan: CoverageSurveyPlan):
    progress = new_survey_progress(plan)
    for viewpoint_id in plan.viewpoint_ids:
        progress = mark_viewpoint_visited(plan, progress, viewpoint_id)
    return progress


def _ranges(target_range: float) -> list[float]:
    result = [3.0] * SCAN_COUNT
    result[TARGET_INDEX] = target_range
    return result


def _receipt(
    receipt_id: str,
    stamp: float,
    target_range: float,
    **overrides,
):
    values = {
        "receipt_id": receipt_id,
        "survey_id": SURVEY_ID,
        "viewpoint_id": CHECK_VIEWPOINT,
        "planning_frame": "map",
        "scan_frame": "base_scan",
        "scan_topic": "/scan",
        "map_bundle_sha256": MAP_SHA256,
        "observer_config_sha256": CONFIG_SHA256,
        "scan_stamp_sec": stamp,
        "pose_stamp_sec": stamp,
        "observer_clock_sec": stamp + 0.01,
        "scan_pose_map": SCAN_POSE,
        "angle_min_rad": ANGLE_MIN,
        "angle_increment_rad": ANGLE_INCREMENT,
        "range_min_m": 0.08,
        "range_max_m": 3.5,
        "ranges_m": _ranges(target_range),
    }
    values.update(overrides)
    return lidar_visibility_receipt_from_scan(**values)


def _clear_receipts():
    return (
        *tuple(
            _receipt(
                f"source_receipt_{index}",
                float(index),
                1.50,
                viewpoint_id=SOURCE_VIEWPOINT,
            )
            for index in range(1, 4)
        ),
        *tuple(
            _receipt(f"check_receipt_{index}", float(index), 1.50)
            for index in range(1, 4)
        ),
    )


def _visibility_evidence(
    *,
    viewpoint_id: str,
    receipts: tuple,
    root: Path,
) -> CoverageVisibilityEvidence:
    return CoverageVisibilityEvidence(
        survey_id=SURVEY_ID,
        viewpoint_id=viewpoint_id,
        planning_frame="map",
        map_bundle_sha256=MAP_SHA256,
        receipts_jsonl=root / f"{viewpoint_id}_receipts.jsonl",
        receipt_count=len(receipts),
        receipts_file_sha256="d" * 64,
        receipt_set_sha256=visibility_receipts_sha256(receipts),
        observer_config={},
        observer_config_sha256=CONFIG_SHA256,
        receipts=receipts,
    )


def _config() -> CoverageCandidateReconciliationConfig:
    return CoverageCandidateReconciliationConfig(
        observer_config_sha256=CONFIG_SHA256,
    )


def _report(*, plan=None, registry=None, receipts=None):
    return build_coverage_candidate_reconciliation_report(
        plan=plan or _plan(),
        registry=registry or _registry(),
        occupancy_grid=_grid(),
        receipts=_clear_receipts() if receipts is None else receipts,
        config=_config(),
    )


class CoverageCandidateReconciliationReportTest(unittest.TestCase):
    def test_clear_decision_is_only_recommended_and_registry_is_unchanged(self):
        registry = _registry()
        snapshot_before = registry

        report = _report(registry=registry)

        self.assertEqual(registry, snapshot_before)
        self.assertEqual(
            report.recommended_negative_visibility_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertEqual(report.decisions[0].action, ACTION_REJECT_PROVISIONAL)
        self.assertFalse(report.registry_mutation_applied)
        self.assertFalse(report.motion_authorized)
        self.assertEqual(report.policy_mode, POLICY_MODE_EVIDENCE_ONLY)
        evidence = report.to_evidence_dict()
        self.assertFalse(evidence["expected_stand_count_used"])
        self.assertFalse(evidence["registry_mutation_applied"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(len(report.report_sha256), 64)
        json.dumps(evidence, allow_nan=False)

    def test_bounded_application_rejects_recommended_provisional_candidate(self):
        registry = _registry()
        report = _report(registry=registry)

        updated, application = apply_negative_visibility_reconciliation_report(
            plan=(plan := _plan()),
            progress=_complete_progress(plan),
            registry=registry,
            report=report,
            included_viewpoint_ids=plan.viewpoint_ids,
        )

        self.assertEqual(registry.candidates[0].status, STATUS_PROVISIONAL)
        self.assertEqual(updated.candidates[0].status, STATUS_REJECTED)
        self.assertEqual(
            updated.candidates[0].rejection_basis,
            REJECTION_BASIS_NEGATIVE_VISIBILITY,
        )
        self.assertEqual(
            application.policy_mode,
            POLICY_MODE_BOUNDED_NEGATIVE_VISIBILITY_REGISTRY_REJECTION,
        )
        self.assertTrue(application.registry_mutation_applied)
        self.assertFalse(application.motion_authorized)
        self.assertEqual(
            application.rejected_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertEqual(
            application.source_registry_snapshot_sha256,
            stand_survey_registry_sha256(registry),
        )
        self.assertEqual(
            application.updated_registry_snapshot_sha256,
            stand_survey_registry_sha256(updated),
        )
        self.assertEqual(len(application.application_sha256), 64)
        json.dumps(application.to_evidence_dict(), allow_nan=False)

    def test_application_rejects_registry_snapshot_mismatch(self):
        report = _report()
        changed_registry = _registry(
            (
                _candidate(),
                _candidate("survey_candidate_0002"),
            )
        )

        with self.assertRaisesRegex(ValueError, "registry snapshot mismatch"):
            apply_negative_visibility_reconciliation_report(
                plan=(plan := _plan()),
                progress=_complete_progress(plan),
                registry=changed_registry,
                report=report,
                included_viewpoint_ids=plan.viewpoint_ids,
            )

    def test_application_is_noop_before_terminal_full_receipt_set(self):
        plan = _plan()
        registry = _registry()
        report = _report(
            plan=plan,
            registry=registry,
            receipts=tuple(
                _receipt(
                    f"source_receipt_{index}",
                    float(index),
                    1.50,
                    viewpoint_id=SOURCE_VIEWPOINT,
                )
                for index in range(1, 4)
            ),
        )
        progress = mark_viewpoint_visited(
            plan,
            new_survey_progress(plan),
            SOURCE_VIEWPOINT,
        )

        updated, application = apply_negative_visibility_reconciliation_report(
            plan=plan,
            progress=progress,
            registry=registry,
            report=report,
            included_viewpoint_ids=(SOURCE_VIEWPOINT,),
        )

        self.assertEqual(updated, registry)
        self.assertFalse(application.terminal_application_eligible)
        self.assertFalse(application.registry_mutation_applied)
        self.assertEqual(
            application.unapplied_recommended_candidate_uids,
            (),
        )
        self.assertIn(
            "planned_viewpoints_incomplete",
            application.application_reasons,
        )

    def test_application_rejects_noncanonical_permissive_config(self):
        plan = _plan()
        registry = _registry()
        report = build_coverage_candidate_reconciliation_report(
            plan=plan,
            registry=registry,
            occupancy_grid=_grid(),
            receipts=_clear_receipts(),
            config=CoverageCandidateReconciliationConfig(
                observer_config_sha256=CONFIG_SHA256,
                minimum_distinct_clear_scan_count=2,
                minimum_clear_ray_fraction=0.0,
                maximum_invalid_selected_ray_fraction=1.0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "non-canonical policy"):
            apply_negative_visibility_reconciliation_report(
                plan=plan,
                progress=_complete_progress(plan),
                registry=registry,
                report=report,
                included_viewpoint_ids=plan.viewpoint_ids,
            )

    def test_application_rejects_tampered_decision_bindings(self):
        plan = _plan()
        registry = _registry()
        report = _report(plan=plan, registry=registry)
        decision = report.decisions[0]
        cases = (
            (
                replace(
                    decision,
                    input_receipt_set_sha256="f" * 64,
                ),
                "receipt-set hash mismatch",
            ),
            (
                replace(
                    decision,
                    source_viewpoint_ids=(CHECK_VIEWPOINT,),
                ),
                "source viewpoint mismatch",
            ),
            (
                replace(decision, action=ACTION_RETAIN),
                "action is inconsistent",
            ),
            (
                replace(
                    decision,
                    ray_policy_decision=replace(
                        decision.ray_policy_decision,
                        rejection_supported=False,
                    ),
                ),
                "ray-policy evidence mismatch",
            ),
        )
        for tampered_decision, message in cases:
            with self.subTest(message=message):
                tampered_report = replace(
                    report,
                    decisions=(tampered_decision,),
                )
                with self.assertRaisesRegex(ValueError, message):
                    apply_negative_visibility_reconciliation_report(
                        plan=plan,
                        progress=_complete_progress(plan),
                        registry=registry,
                        report=tampered_report,
                        included_viewpoint_ids=plan.viewpoint_ids,
                    )

    def test_six_to_five_projection_is_consistent_through_camera_snapshot(self):
        plan = _plan(expected_count=5)
        registry = _registry(
            (
                _candidate("survey_candidate_0001"),
                *tuple(
                    _candidate(
                        f"survey_candidate_{index:04d}",
                        viewpoint_ids=(SOURCE_VIEWPOINT, CHECK_VIEWPOINT),
                        status=STATUS_PENDING_CAMERA,
                    )
                    for index in range(2, 7)
                ),
            )
        )
        progress = _complete_progress(plan)
        report = _report(plan=plan, registry=registry)
        before = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)

        updated, application = apply_negative_visibility_reconciliation_report(
            plan=plan,
            progress=progress,
            registry=registry,
            report=report,
            included_viewpoint_ids=plan.viewpoint_ids,
        )
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, updated)
        camera = evaluate_exact_two_camera_admission(
            plan,
            progress,
            updated,
            lidar,
        )
        snapshot = build_exact_two_camera_candidate_snapshot(
            plan,
            updated,
            camera,
            snapshot_id="reconciled_candidates",
        )

        self.assertFalse(before.ready)
        self.assertIn(
            "strict_candidate_count_exceeds_expected",
            before.reasons,
        )
        self.assertEqual(
            application.rejected_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertTrue(lidar.ready)
        self.assertTrue(camera.ready)
        self.assertEqual(len(snapshot.candidates), 5)
        self.assertEqual(
            camera.source_registry_sha256,
            stand_survey_registry_sha256(updated),
        )

    def test_terminal_checkpoint_and_camera_handoff_bind_updated_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve(strict=True)
            survey_root = root / "coverage"
            plan = _plan(expected_count=5)
            registry = _registry(
                (
                    _candidate("survey_candidate_0001"),
                    *tuple(
                        _candidate(
                            f"survey_candidate_{index:04d}",
                            viewpoint_ids=(SOURCE_VIEWPOINT, CHECK_VIEWPOINT),
                            status=STATUS_PENDING_CAMERA,
                        )
                        for index in range(2, 7)
                    ),
                )
            )
            prior_progress = mark_viewpoint_visited(
                plan,
                new_survey_progress(plan),
                SOURCE_VIEWPOINT,
            )
            completed_progress = mark_viewpoint_visited(
                plan,
                prior_progress,
                CHECK_VIEWPOINT,
            )
            receipts = _clear_receipts()
            source_evidence = _visibility_evidence(
                viewpoint_id=SOURCE_VIEWPOINT,
                receipts=tuple(
                    receipt
                    for receipt in receipts
                    if receipt.viewpoint_id == SOURCE_VIEWPOINT
                ),
                root=root,
            )
            current_evidence = _visibility_evidence(
                viewpoint_id=CHECK_VIEWPOINT,
                receipts=tuple(
                    receipt
                    for receipt in receipts
                    if receipt.viewpoint_id == CHECK_VIEWPOINT
                ),
                root=root,
            )

            with patch(
                "scripts.aufgabe04.navigation.coverage."
                "coverage_stop_perception_admission."
                "_load_validated_visibility_epochs",
                return_value=(source_evidence, current_evidence),
            ):
                reconciliation = prepare_coverage_visibility_reconciliation(
                    survey_root=survey_root,
                    plan=plan,
                    prior_progress=prior_progress,
                    completed_progress=completed_progress,
                    current_viewpoint_id=CHECK_VIEWPOINT,
                    current_evidence=current_evidence,
                    registry=registry,
                    occupancy_grid=_grid(),
                )

            self.assertIsNotNone(reconciliation)
            assert reconciliation is not None
            self.assertEqual(
                reconciliation.application.rejected_candidate_uids,
                ("survey_candidate_0001",),
            )
            for artifact in reconciliation.evidence_artifacts:
                artifact.path.parent.mkdir(parents=True, exist_ok=True)
                self.assertEqual(
                    write_content_hashed_json(
                        artifact.path,
                        artifact.payload,
                        hash_field=artifact.hash_field,
                    ),
                    artifact.sha256,
                )

            plan_path = survey_root / "coverage_plan.json"
            progress_path = survey_root / "coverage_progress.json"
            registry_path = survey_root / "stand_registry.json"
            summary_path = survey_root / "survey_summary.json"
            observer_path = root / "lidar_observer_summary.json"
            write_coverage_survey_plan(plan_path, plan)
            write_survey_progress(progress_path, completed_progress, plan)
            write_stand_survey_registry(
                registry_path,
                reconciliation.updated_registry,
                plan,
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "motion_authorized": False,
                        "lidar_visibility_reconciliation_sha256": (
                            reconciliation.artifact.sha256
                        ),
                        "lidar_visibility_reconciliation_json": str(
                            reconciliation.artifact.path
                        ),
                        "lidar_visibility_reconciliation_application_sha256": (
                            reconciliation.application_artifact.sha256
                        ),
                        "lidar_visibility_reconciliation_application_json": str(
                            reconciliation.application_artifact.path
                        ),
                        "stand_registry_sha256": stand_survey_registry_sha256(
                            reconciliation.updated_registry
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            observer_path.write_text('{"motion_published": false}\n')
            session_root = root / "session"
            session_root.mkdir()
            published = publish_coverage_checkpoint(
                session_root=session_root,
                session_id="terminal_reconciliation",
                run_mode="execute-exact-two-camera",
                robot_id="tb3_1",
                robot_profile_sha256="a" * 64,
                calibration_profile_sha256="b" * 64,
                physical_site_sha256="c" * 64,
                map_bundle_sha256=plan.map_bundle_sha256,
                config_sha256="e" * 64,
                completed_coverage_legs=2,
                next_viewpoint_id=None,
                coverage_plan_path=plan_path,
                coverage_progress_path=progress_path,
                survey_summary_path=summary_path,
                stand_registry_path=registry_path,
                lidar_observer_summary_path=observer_path,
                status=COVERAGE_SURVEY_TERMINAL_CHECKPOINT,
            )

            admitted_manifest = admit_autonomous_session_manifest(
                published.manifest_path
            )
            checkpoint_plan = load_coverage_survey_plan(
                Path(admitted_manifest.coverage_plan.path)
            )
            checkpoint_progress = load_survey_progress(
                Path(admitted_manifest.coverage_progress.path),
                checkpoint_plan,
            )
            checkpoint_registry_path = Path(admitted_manifest.stand_registry.path)
            self.assertEqual(
                hashlib.sha256(checkpoint_registry_path.read_bytes()).hexdigest(),
                admitted_manifest.stand_registry.sha256,
            )
            checkpoint_registry = load_stand_survey_registry(
                checkpoint_registry_path,
                checkpoint_plan,
            )
            checkpoint_summary = json.loads(
                Path(admitted_manifest.survey_summary.path).read_text()
            )
            report_artifact = load_content_hashed_json(
                Path(checkpoint_summary["lidar_visibility_reconciliation_json"]),
                hash_field="lidar_visibility_reconciliation_sha256",
            )
            application_artifact = load_content_hashed_json(
                Path(
                    checkpoint_summary[
                        "lidar_visibility_reconciliation_application_json"
                    ]
                ),
                hash_field=(
                    "lidar_visibility_reconciliation_application_sha256"
                ),
            )
            rejected = checkpoint_registry.candidates[0]
            self.assertEqual(
                checkpoint_registry.schema_version,
                STAND_SURVEY_REGISTRY_SCHEMA_VERSION,
            )
            self.assertEqual(rejected.status, STATUS_REJECTED)
            self.assertEqual(
                rejected.rejection_basis,
                REJECTION_BASIS_NEGATIVE_VISIBILITY,
            )
            self.assertEqual(
                stand_survey_registry_sha256(checkpoint_registry),
                reconciliation.application.updated_registry_snapshot_sha256,
            )
            self.assertEqual(
                report_artifact,
                reconciliation.artifact.payload,
            )
            self.assertEqual(
                checkpoint_summary["lidar_visibility_reconciliation_sha256"],
                reconciliation.artifact.sha256,
            )
            self.assertEqual(
                application_artifact,
                reconciliation.application_artifact.payload,
            )
            self.assertEqual(
                checkpoint_summary[
                    "lidar_visibility_reconciliation_application_sha256"
                ],
                reconciliation.application_artifact.sha256,
            )

            lidar = evaluate_exact_two_lidar_checkpoint(
                checkpoint_plan,
                checkpoint_progress,
                checkpoint_registry,
            )
            camera = evaluate_exact_two_camera_admission(
                checkpoint_plan,
                checkpoint_progress,
                checkpoint_registry,
                lidar,
            )
            snapshot = build_exact_two_camera_candidate_snapshot(
                checkpoint_plan,
                checkpoint_registry,
                camera,
                snapshot_id="terminal_reconciled_candidates",
            )
            handoff = new_exact_two_camera_handoff(
                handoff_id="terminal_reconciliation_handoff",
                created_unix_sec=10.0,
                admission=camera,
                terminal_checkpoint_path=published.manifest_path,
                terminal_checkpoint_sha256=published.manifest_sha256,
                lidar_admission_path=root / "lidar_admission.json",
                lidar_admission_sha256=camera.lidar_checkpoint_sha256,
                camera_admission_path=root / "camera_admission.json",
                camera_admission_sha256=exact_two_camera_admission_sha256(
                    camera
                ),
                candidate_snapshot_path=root / "candidate_snapshot.json",
                candidate_snapshot=snapshot,
            )
            validate_live_registry_binding(handoff, checkpoint_registry)
            with self.assertRaisesRegex(ValueError, "live stand registry"):
                validate_live_registry_binding(handoff, registry)

            self.assertTrue(lidar.ready)
            self.assertTrue(camera.ready)
            self.assertEqual(len(snapshot.candidates), 5)
            self.assertEqual(
                camera.source_registry_sha256,
                stand_survey_registry_sha256(checkpoint_registry),
            )

    def test_missing_or_invalid_receipts_retain_provisional_candidate(self):
        missing = _report(receipts=())
        self.assertEqual(
            missing.retained_provisional_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertEqual(missing.decisions[0].action, ACTION_RETAIN)
        self.assertIn(
            "visibility_receipts_missing",
            missing.decisions[0].reasons,
        )

        invalid_for_config = tuple(
            _receipt(
                f"receipt_{index}",
                float(index),
                math.inf,
                observer_config_sha256="c" * 64,
            )
            for index in range(1, 4)
        )
        invalid = _report(receipts=invalid_for_config)
        self.assertEqual(
            invalid.recommended_negative_visibility_candidate_uids,
            (),
        )
        self.assertEqual(
            invalid.retained_provisional_candidate_uids,
            ("survey_candidate_0001",),
        )
        self.assertIn(
            "visibility_receipt_identity_mismatch",
            invalid.decisions[0].reasons,
        )
        self.assertIn(
            "selected_scan_ray_invalid",
            invalid.decisions[0].reasons,
        )

    def test_recommendations_are_independent_of_expected_count(self):
        expected_five = _report(plan=_plan(expected_count=5))
        expected_ninety_nine = _report(plan=_plan(expected_count=99))

        self.assertEqual(
            expected_five.recommended_negative_visibility_candidate_uids,
            expected_ninety_nine.recommended_negative_visibility_candidate_uids,
        )
        self.assertEqual(
            tuple(decision.action for decision in expected_five.decisions),
            tuple(
                decision.action for decision in expected_ninety_nine.decisions
            ),
        )
        self.assertEqual(
            expected_five.registry_snapshot_sha256,
            expected_ninety_nine.registry_snapshot_sha256,
        )

    def test_candidates_and_receipts_have_stable_canonical_order(self):
        registry = _registry(
            (
                _candidate("survey_candidate_0001"),
                _candidate("survey_candidate_0002"),
                _candidate(
                    "survey_candidate_0003",
                    viewpoint_ids=(SOURCE_VIEWPOINT, CHECK_VIEWPOINT),
                ),
            )
        )
        receipts = _clear_receipts()

        forward = _report(registry=registry, receipts=receipts)
        reversed_receipts = _report(
            registry=registry,
            receipts=tuple(reversed(receipts)),
        )

        self.assertEqual(
            tuple(decision.candidate_uid for decision in forward.decisions),
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertEqual(
            forward.recommended_negative_visibility_candidate_uids,
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertEqual(
            forward.retained_provisional_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertEqual(
            forward.unevaluated_provisional_candidate_uids,
            ("survey_candidate_0003",),
        )
        self.assertEqual(
            forward.receipt_set_sha256,
            reversed_receipts.receipt_set_sha256,
        )
        self.assertEqual(forward.report_sha256, reversed_receipts.report_sha256)


if __name__ == "__main__":
    unittest.main()
