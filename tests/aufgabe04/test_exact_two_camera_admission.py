from __future__ import annotations

import ast
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.approach.exact_two_camera_admission import (
    SOURCE_KIND_MULTI_VIEW,
    SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    SUPPORT_CLASS_MULTI_VIEW,
    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
    ExactTwoCameraAdmissionError,
    build_exact_two_camera_candidate_snapshot,
    evaluate_exact_two_camera_admission,
    exact_two_camera_admission_payload,
    exact_two_camera_admission_sha256,
    exact_two_camera_handoff_payload,
    exact_two_camera_handoff_sha256,
    load_bound_exact_two_candidate_snapshot,
    load_exact_two_camera_admission,
    load_exact_two_camera_handoff,
    new_exact_two_camera_handoff,
    require_admitted_candidate_support,
    require_handoff_candidate_support,
    stand_survey_registry_sha256,
    validate_exact_two_camera_admission,
    validate_live_candidate_snapshot_binding,
    validate_live_registry_binding,
    write_exact_two_camera_admission,
    write_exact_two_camera_handoff,
)
from scripts.aufgabe04.navigation.coverage.coverage_candidate_lifecycle import (
    evaluate_exact_two_lidar_checkpoint,
)
from scripts.aufgabe04.navigation.foundation.models import GridCell, Pose2D
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
    STATUS_PROVISIONAL,
    SURVEY_PLAN_SCHEMA_VERSION,
    CoverageSurveyConfig,
    CoverageSurveyPlan,
    StandSurveyRegistry,
    SurveyCandidate,
    SurveyViewpoint,
    mark_viewpoint_visited,
    new_stand_survey_registry,
    new_survey_progress,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256,
    write_candidate_snapshot,
)


MAP_HASH = "a" * 64
TERMINAL_HASH = "b" * 64
LIDAR_WRAPPER_HASH = "c" * 64


def _plan(*, expected_stand_count: int | None = 5) -> CoverageSurveyPlan:
    cells = tuple(GridCell(index, 0) for index in range(4))
    return CoverageSurveyPlan(
        schema_version=SURVEY_PLAN_SCHEMA_VERSION,
        survey_id="exact_two_camera_test",
        planning_frame="map",
        map_bundle_sha256=MAP_HASH,
        arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
        config=CoverageSurveyConfig(
            lane_count=1,
            coverage_threshold=0.75,
            minimum_candidate_confidence=0.70,
            minimum_candidate_hits=3,
            minimum_distinct_viewpoints=2,
            expected_stand_count=expected_stand_count,
            exact_inspection_point_count=2,
        ),
        viewpoints=(
            SurveyViewpoint(
                viewpoint_id="survey_vp_001",
                pose=Pose2D(-1.0, 0.0, 0.0),
                cell=cells[0],
                visible_cells=cells[:3],
            ),
            SurveyViewpoint(
                viewpoint_id="survey_vp_002",
                pose=Pose2D(1.0, 0.0, 0.0),
                cell=cells[-1],
                visible_cells=cells[1:],
            ),
        ),
        surveyable_cells=cells,
        planned_covered_cells=cells,
        planned_coverage_ratio=1.0,
    )


def _complete_progress(plan: CoverageSurveyPlan):
    progress = new_survey_progress(plan)
    for viewpoint_id in plan.viewpoint_ids:
        progress = mark_viewpoint_visited(plan, progress, viewpoint_id)
    return progress


def _candidate(
    index: int,
    *,
    status: str = STATUS_PROVISIONAL,
    confidence: float = 0.82,
    hit_count: int = 7,
    viewpoint_ids: tuple[str, ...] = ("survey_vp_001",),
) -> SurveyCandidate:
    return SurveyCandidate(
        candidate_uid=f"survey_candidate_{index:04d}",
        x_m=0.25 * index,
        y_m=0.10 * index,
        radius_m=0.06,
        uncertainty_m=0.02,
        keepout_radius_m=0.31,
        confidence=confidence,
        hit_count=hit_count,
        first_seen_sec=10.0 + index,
        last_seen_sec=11.0 + index,
        source_observation_ids=(f"observation_{index:04d}",),
        viewpoint_ids=viewpoint_ids,
        status=status,
    )


def _registry(
    plan: CoverageSurveyPlan, *candidates: SurveyCandidate
) -> StandSurveyRegistry:
    return replace(
        new_stand_survey_registry(plan),
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_uid)),
    )


def _latest_run_registry(plan: CoverageSurveyPlan) -> StandSurveyRegistry:
    both = ("survey_vp_001", "survey_vp_002")
    return _registry(
        plan,
        _candidate(1, status=STATUS_PENDING_CAMERA, viewpoint_ids=both),
        _candidate(2, status=STATUS_PENDING_CAMERA, viewpoint_ids=both),
        _candidate(3),
        _candidate(4, viewpoint_ids=("survey_vp_002",)),
        _candidate(5),
    )


def _ready_inputs():
    plan = _plan()
    progress = _complete_progress(plan)
    registry = _latest_run_registry(plan)
    lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
    admission = evaluate_exact_two_camera_admission(
        plan, progress, registry, lidar
    )
    return plan, progress, registry, lidar, admission


class ExactTwoCameraAdmissionDecisionTest(unittest.TestCase):
    def test_module_slice_is_ros_free_and_does_not_import_parent_runner(self):
        root = Path(__file__).resolve().parents[2]
        module_paths = (
            root / "scripts/aufgabe04/navigation/approach/exact_two_camera_admission.py",
            root / "scripts/aufgabe04/navigation/approach/exact_two_camera_contract.py",
            root / "scripts/aufgabe04/navigation/approach/exact_two_camera_artifacts.py",
        )
        forbidden_roots = {"rclpy", "subprocess"}
        contract_types = {
            "ExactTwoCameraCandidateEvidence",
            "ExactTwoCameraAdmissionDecision",
            "ExactTwoCameraHandoffArtifact",
        }
        for path in module_paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(
                any(name.split(".", 1)[0] in forbidden_roots for name in imports),
                path,
            )
            self.assertFalse(
                any("scripts.aufgabe04.real_robot" in name for name in imports),
                path,
            )
            self.assertFalse(
                any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "input"
                    for node in ast.walk(tree)
                ),
                path,
            )
            defined_classes = {
                node.name for node in tree.body if isinstance(node, ast.ClassDef)
            }
            if path.name == "exact_two_camera_contract.py":
                self.assertTrue(contract_types.issubset(defined_classes))
            else:
                self.assertTrue(contract_types.isdisjoint(defined_classes))
            self.assertLess(len(path.read_text().splitlines()), 800, path)

    def test_latest_run_shape_admits_two_multi_view_and_three_single_view(self):
        plan, progress, registry, lidar, decision = _ready_inputs()

        self.assertTrue(lidar.ready)
        self.assertTrue(decision.ready)
        self.assertTrue(decision.camera_population_ready)
        self.assertFalse(decision.motion_authorized)
        self.assertEqual(
            decision.multi_view_candidate_uids,
            ("survey_candidate_0001", "survey_candidate_0002"),
        )
        self.assertEqual(
            decision.single_view_candidate_uids,
            (
                "survey_candidate_0003",
                "survey_candidate_0004",
                "survey_candidate_0005",
            ),
        )
        self.assertEqual(decision.blocked_candidate_uids, ())
        self.assertEqual(len(decision.admitted_candidate_uids), 5)
        self.assertEqual(registry, _latest_run_registry(plan))
        self.assertEqual(progress, _complete_progress(plan))

        direct = require_admitted_candidate_support(
            decision,
            "survey_candidate_0001",
            SUPPORT_CLASS_MULTI_VIEW,
        )
        single = require_admitted_candidate_support(
            decision,
            "survey_candidate_0003",
            SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
        )
        self.assertEqual(direct.source_kind, SOURCE_KIND_MULTI_VIEW)
        self.assertEqual(
            single.source_kind,
            SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
        )

    def test_forged_static_map_admission_cannot_enter_ready_population(self):
        *_, decision = _ready_inputs()
        forged = replace(
            decision,
            candidate_evidence=(
                replace(
                    decision.candidate_evidence[0],
                    static_map_admitted=False,
                ),
                *decision.candidate_evidence[1:],
            ),
        )
        with self.assertRaises(ExactTwoCameraAdmissionError) as raised:
            validate_exact_two_camera_admission(forged)
        self.assertEqual(raised.exception.code, "invalid_admission")

    def test_snapshot_projection_uses_standard_type_and_support_sources(self):
        plan, _, registry, _, decision = _ready_inputs()
        snapshot = build_exact_two_camera_candidate_snapshot(
            plan, registry, decision, snapshot_id="camera_candidates_001"
        )

        self.assertEqual(snapshot.candidate_uids, decision.admitted_candidate_uids)
        self.assertEqual(
            snapshot.candidates[0].source.source_kind,
            SOURCE_KIND_MULTI_VIEW,
        )
        self.assertEqual(
            snapshot.candidates[2].source.source_kind,
            SOURCE_KIND_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
        )
        self.assertTrue(
            all(
                item.source.source_artifact_sha256
                == decision.source_registry_sha256
                for item in snapshot.candidates
            )
        )

    def test_weak_unknown_and_non_single_provisional_are_blocked(self):
        plan = _plan()
        progress = _complete_progress(plan)
        both = plan.viewpoint_ids
        registry = _registry(
            plan,
            _candidate(1, status=STATUS_PENDING_CAMERA, viewpoint_ids=both),
            _candidate(2, confidence=0.69),
            _candidate(3, viewpoint_ids=("unknown_vp",)),
            _candidate(4, viewpoint_ids=both),
            _candidate(5),
        )
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
        decision = evaluate_exact_two_camera_admission(
            plan, progress, registry, lidar
        )

        self.assertFalse(decision.ready)
        self.assertFalse(lidar.ready)
        self.assertEqual(
            decision.blocked_candidate_uids,
            (
                "survey_candidate_0002",
                "survey_candidate_0003",
                "survey_candidate_0004",
            ),
        )
        self.assertIn("lidar_checkpoint_not_ready", decision.reasons)
        self.assertIn(
            "active_candidates_not_camera_admissible", decision.reasons
        )
        self.assertIn(
            "confidence_below_minimum",
            decision.candidate_for("survey_candidate_0002").reasons,
        )
        self.assertIn(
            "unknown_viewpoint_ids",
            decision.candidate_for("survey_candidate_0003").reasons,
        )
        self.assertIn(
            "provisional_candidate_not_single_view",
            decision.candidate_for("survey_candidate_0004").reasons,
        )

    def test_count_mismatch_blocks_population_even_when_each_candidate_is_strong(self):
        plan = _plan(expected_stand_count=5)
        progress = _complete_progress(plan)
        registry = _registry(plan, *_latest_run_registry(plan).candidates[:4])
        lidar = evaluate_exact_two_lidar_checkpoint(plan, progress, registry)
        decision = evaluate_exact_two_camera_admission(
            plan, progress, registry, lidar
        )

        self.assertFalse(decision.ready)
        self.assertEqual(decision.blocked_candidate_uids, ())
        self.assertEqual(
            decision.reasons,
            ("lidar_checkpoint_not_ready", "active_candidate_count_mismatch"),
        )
        self.assertEqual(decision.admitted_candidate_uids, ())
        with self.assertRaisesRegex(
            ExactTwoCameraAdmissionError, "not-ready admission"
        ):
            build_exact_two_camera_candidate_snapshot(
                plan, registry, decision, snapshot_id="not_ready"
            )

    def test_substituted_checkpoint_is_rejected_not_reinterpreted(self):
        plan, progress, registry, lidar, _ = _ready_inputs()
        substituted = replace(lidar, registry_snapshot_sha256="f" * 64)
        with self.assertRaises(ExactTwoCameraAdmissionError) as raised:
            evaluate_exact_two_camera_admission(
                plan, progress, registry, substituted
            )
        self.assertEqual(raised.exception.code, "provenance_mismatch")

    def test_hash_and_order_are_deterministic_and_values_are_frozen(self):
        plan, progress, registry, lidar, first = _ready_inputs()
        second = evaluate_exact_two_camera_admission(
            plan, progress, registry, lidar
        )
        payload = first.to_evidence_dict()
        json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertEqual(first, second)
        self.assertEqual(
            exact_two_camera_admission_sha256(first),
            exact_two_camera_admission_sha256(payload),
        )
        self.assertEqual(
            stand_survey_registry_sha256(registry),
            lidar.registry_snapshot_sha256,
        )
        with self.assertRaises(FrozenInstanceError):
            first.ready = False
        with self.assertRaises(FrozenInstanceError):
            first.candidate_evidence[0].admissible = False

        reversed_registry = replace(
            registry, candidates=tuple(reversed(registry.candidates))
        )
        with self.assertRaises(ExactTwoCameraAdmissionError):
            stand_survey_registry_sha256(reversed_registry)

    def test_support_lookup_rejects_unknown_uid_and_class_substitution(self):
        *_, decision = _ready_inputs()
        with self.assertRaises(ExactTwoCameraAdmissionError) as missing:
            require_admitted_candidate_support(decision, "survey_candidate_9999")
        self.assertEqual(missing.exception.code, "candidate_not_admitted")
        with self.assertRaises(ExactTwoCameraAdmissionError) as wrong:
            require_admitted_candidate_support(
                decision,
                "survey_candidate_0003",
                SUPPORT_CLASS_MULTI_VIEW,
            )
        self.assertEqual(wrong.exception.code, "support_class_mismatch")


class ExactTwoCameraArtifactTest(unittest.TestCase):
    def _artifacts(self, root: Path):
        plan, _, registry, _, admission = _ready_inputs()
        admission_path = root / "coverage_exact_two_camera_admission.json"
        admission_hash = write_exact_two_camera_admission(
            admission_path, admission
        )
        snapshot_path = root / "candidate_snapshot.json"
        snapshot = build_exact_two_camera_candidate_snapshot(
            plan, registry, admission, snapshot_id="camera_candidates_001"
        )
        write_candidate_snapshot(snapshot_path, snapshot)
        handoff = new_exact_two_camera_handoff(
            handoff_id="exact_two_camera_handoff_001",
            created_unix_sec=100.0,
            admission=admission,
            terminal_checkpoint_path=root / "mission_summary.json",
            terminal_checkpoint_sha256=TERMINAL_HASH,
            lidar_admission_path=root / "coverage_lidar_checkpoint_admission.json",
            lidar_admission_sha256=LIDAR_WRAPPER_HASH,
            camera_admission_path=admission_path,
            camera_admission_sha256=admission_hash,
            candidate_snapshot_path=snapshot_path,
            candidate_snapshot=snapshot,
        )
        return plan, registry, admission_path, snapshot_path, snapshot, handoff

    def test_admission_and_handoff_round_trip_bind_every_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (
                _,
                registry,
                admission_path,
                snapshot_path,
                snapshot,
                handoff,
            ) = self._artifacts(root)
            handoff_path = root / "coverage_exact_two_camera_handoff.json"
            written = write_exact_two_camera_handoff(handoff_path, handoff)
            loaded = load_exact_two_camera_handoff(handoff_path)

            self.assertEqual(load_exact_two_camera_admission(admission_path), handoff.admission_decision)
            self.assertEqual(loaded, handoff)
            self.assertEqual(written, exact_two_camera_handoff_sha256(handoff))
            self.assertEqual(
                handoff.candidate_snapshot_sha256,
                candidate_snapshot_sha256(snapshot),
            )
            self.assertFalse(handoff.motion_authorized)
            self.assertTrue(handoff.camera_population_ready)
            self.assertEqual(handoff.lidar_admission_sha256, LIDAR_WRAPPER_HASH)
            self.assertEqual(
                handoff.lidar_checkpoint_sha256,
                handoff.admission_decision.lidar_checkpoint_sha256,
            )
            self.assertNotEqual(
                handoff.lidar_admission_sha256,
                handoff.lidar_checkpoint_sha256,
            )
            validate_live_registry_binding(loaded, registry)
            validate_live_candidate_snapshot_binding(
                loaded, snapshot, candidate_snapshot_path=snapshot_path
            )
            self.assertEqual(
                load_bound_exact_two_candidate_snapshot(loaded, snapshot_path),
                snapshot,
            )
            self.assertEqual(
                require_handoff_candidate_support(
                    loaded,
                    "survey_candidate_0004",
                    SUPPORT_CLASS_SINGLE_VIEW_REQUIRES_CAMERA_VALIDATION,
                ).candidate_uid,
                "survey_candidate_0004",
            )

    def test_root_hash_and_nested_admission_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            *_, handoff = self._artifacts(root)
            path = root / "handoff.json"
            write_exact_two_camera_handoff(path, handoff)
            payload = json.loads(path.read_text())
            payload["candidate_snapshot_sha256"] = "f" * 64
            path.write_text(json.dumps(payload))
            with self.assertRaises(ExactTwoCameraAdmissionError) as root_tamper:
                load_exact_two_camera_handoff(path)
            self.assertEqual(root_tamper.exception.code, "hash_mismatch")

            payload = exact_two_camera_handoff_payload(handoff)
            del payload["exact_two_camera_handoff_sha256"]
            payload["admission_decision"]["motion_authorized"] = True
            payload["exact_two_camera_handoff_sha256"] = payload_sha256(payload)
            path.write_text(json.dumps(payload))
            with self.assertRaises(ExactTwoCameraAdmissionError) as nested:
                load_exact_two_camera_handoff(path)
            self.assertEqual(nested.exception.code, "motion_scope_violation")

    def test_unknown_fields_and_reordered_uid_lists_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            *_, handoff = self._artifacts(root)
            path = root / "handoff.json"
            payload = exact_two_camera_handoff_payload(handoff)
            del payload["exact_two_camera_handoff_sha256"]
            payload["unexpected"] = True
            payload["exact_two_camera_handoff_sha256"] = payload_sha256(payload)
            path.write_text(json.dumps(payload))
            with self.assertRaises(ExactTwoCameraAdmissionError) as unknown:
                load_exact_two_camera_handoff(path)
            self.assertEqual(unknown.exception.code, "artifact_corrupt")

            admission = handoff.admission_decision
            admission_payload = exact_two_camera_admission_payload(admission)
            del admission_payload["exact_two_camera_admission_sha256"]
            admission_payload["multi_view_candidate_uids"].reverse()
            admission_payload["exact_two_camera_admission_sha256"] = payload_sha256(
                admission_payload
            )
            admission_path = root / "reordered_admission.json"
            admission_path.write_text(json.dumps(admission_payload))
            with self.assertRaises(ExactTwoCameraAdmissionError) as reordered:
                load_exact_two_camera_admission(admission_path)
            self.assertEqual(reordered.exception.code, "invalid_admission")

    def test_immutable_publication_and_path_hash_substitution_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            *_, handoff = self._artifacts(root)
            path = root / "handoff.json"
            first = write_exact_two_camera_handoff(path, handoff)
            retry = write_exact_two_camera_handoff(path, handoff)
            self.assertEqual(first, retry)
            with self.assertRaises(ExactTwoCameraAdmissionError) as conflict:
                write_exact_two_camera_handoff(
                    path, replace(handoff, handoff_id="different_handoff")
                )
            self.assertEqual(conflict.exception.code, "immutable_conflict")

            with self.assertRaises(ExactTwoCameraAdmissionError) as lidar:
                new_exact_two_camera_handoff(
                    handoff_id="bad_lidar",
                    created_unix_sec=100.0,
                    admission=handoff.admission_decision,
                    terminal_checkpoint_path=handoff.terminal_checkpoint_path,
                    terminal_checkpoint_sha256=TERMINAL_HASH,
                    lidar_admission_path=handoff.lidar_admission_path,
                    lidar_admission_sha256="not-a-sha256",
                    camera_admission_path=handoff.camera_admission_path,
                    camera_admission_sha256=handoff.camera_admission_sha256,
                    candidate_snapshot_path=handoff.candidate_snapshot_path,
                    candidate_snapshot=load_bound_exact_two_candidate_snapshot(
                        handoff, Path(handoff.candidate_snapshot_path)
                    ),
                )
            self.assertEqual(lidar.exception.code, "invalid_hash")

    def test_live_registry_and_snapshot_substitution_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, registry, _, snapshot_path, snapshot, handoff = self._artifacts(root)
            changed_registry = replace(
                registry,
                candidates=(
                    replace(registry.candidates[0], confidence=0.83),
                    *registry.candidates[1:],
                ),
            )
            with self.assertRaises(ExactTwoCameraAdmissionError) as stale:
                validate_live_registry_binding(handoff, changed_registry)
            self.assertEqual(stale.exception.code, "live_registry_mismatch")

            moved = replace(
                snapshot,
                candidates=(
                    replace(
                        snapshot.candidates[0],
                        geometry=replace(
                            snapshot.candidates[0].geometry,
                            x_m=snapshot.candidates[0].geometry.x_m + 0.01,
                        ),
                    ),
                    *snapshot.candidates[1:],
                ),
            )
            with self.assertRaises(ExactTwoCameraAdmissionError) as changed:
                validate_live_candidate_snapshot_binding(handoff, moved)
            self.assertEqual(changed.exception.code, "live_snapshot_mismatch")
            with self.assertRaises(ExactTwoCameraAdmissionError) as wrong_path:
                validate_live_candidate_snapshot_binding(
                    handoff,
                    snapshot,
                    candidate_snapshot_path=snapshot_path.with_name("other.json"),
                )
            self.assertEqual(wrong_path.exception.code, "live_snapshot_mismatch")


if __name__ == "__main__":
    unittest.main()
