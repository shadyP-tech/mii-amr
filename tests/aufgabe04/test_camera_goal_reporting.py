from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.navigation.coverage.candidate_inspection_pool import (
    candidate_inspection_pool_policy_evidence,
)
from scripts.aufgabe04.real_robot.mission.camera_goal_reporting import validate_completed_qr_goal
from scripts.aufgabe04.real_robot.mission.exact_two_completion import _require_exact_two_snapshot_population
from scripts.aufgabe04.real_robot.mission.reporting import build_completed_camera_mission_summary


def fields():
    return {
        "goal_completed": True, "stand_count": 5, "expected_stand_count": 5,
        "candidate_pool_count": 6, "candidate_snapshot_sha256": "a" * 64,
        "confirmed_candidate_uids": [f"candidate_{i}" for i in range(5)],
        "remaining_candidate_uids": ["candidate_5"],
        "candidate_goal_progress": "progress/revision_005.json",
        "candidate_goal_progress_sha256": "b" * 64,
        "confirmed_candidate_snapshot": "confirmed.json",
        "confirmed_candidate_snapshot_sha256": "c" * 64,
        "motion_authorized": False,
    }


def coverage():
    uids = [f"candidate_{i}" for i in range(6)]
    return {
        "expected_stand_count": 5,
        "inspection_pool_policy": candidate_inspection_pool_policy_evidence(5),
        "camera_seed_candidate_uids": uids,
        "camera_validation_candidate_uids": uids,
        "camera_seed_candidate_count": 6,
        "camera_seed_selection_mode": "bounded_inspection_pool",
        "active_lidar_registry_candidate_count": 6,
        "lidar_static_map_admitted_candidate_count": 6,
        "lidar_boundary_provisional_candidate_count": 0,
        "lidar_population_retained_candidate_count": 6,
        "camera_seed_boundary_fill_candidate_uids": [],
        "camera_seed_boundary_audit_only_candidate_uids": [],
        "camera_seed_excluded_candidate_uids": [],
        "multi_view_candidate_uids": uids[:2],
        "single_view_requires_camera_validation_candidate_uids": uids[2:],
        "lidar_checkpoint_admission": "lidar.json",
        "lidar_checkpoint_admission_sha256": "d" * 64,
        "camera_validation_admission": "camera.json",
        "camera_validation_admission_sha256": "e" * 64,
    }


class CameraGoalReportingTests(unittest.TestCase):
    def test_qr_fallback_finishes_exploration_without_claiming_complete_geometry(self):
        discovery = {**fields(), "facing_ready_stand_count": 3, "qr_only_stand_count": 2,
                     "facing_complete": False, "qr_observation_pose_catalog": "qr_poses.json",
                     "qr_observation_pose_catalog_sha256": "f" * 64}
        result = build_completed_camera_mission_summary(
            run_mode="execute-exact-two-camera", session_id="session_1",
            snapshot_path=Path("full_pool.json"), snapshot_sha256="a" * 64,
            survey_root=Path("survey"), stand_model_profile=Path("model.json"),
            stand_model_profile_sha256="f" * 64,
            candidate_population_admission_path=Path("admission.json"),
            candidate_population_admission_sha256="d" * 64,
            candidate_phase_fields=discovery, exact_two_coverage_summary=coverage(),
            exact_two_camera_handoff_path=Path("handoff.json"),
            exact_two_camera_handoff_sha256="e" * 64,
        )
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["camera_exploration_complete"])
        self.assertFalse(result["camera_geometry_complete"])
        self.assertFalse(result["facing_complete"])
        self.assertEqual(result["stand_count"], 5)
        self.assertEqual(result["qr_only_stand_count"], 2)
        self.assertFalse(result["motion_authorized"])

    def test_five_confirmed_from_six_pool_preserves_both_snapshots(self):
        result = build_completed_camera_mission_summary(
            run_mode="execute-exact-two-camera", session_id="session_1",
            snapshot_path=Path("full_pool.json"), snapshot_sha256="a" * 64,
            survey_root=Path("survey"), stand_model_profile=Path("model.json"),
            stand_model_profile_sha256="f" * 64,
            candidate_population_admission_path=Path("admission.json"),
            candidate_population_admission_sha256="d" * 64,
            candidate_phase_fields=fields(), exact_two_coverage_summary=coverage(),
            exact_two_camera_handoff_path=Path("handoff.json"),
            exact_two_camera_handoff_sha256="e" * 64,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["stand_count"], 5)
        self.assertEqual(result["candidate_pool_count"], 6)
        self.assertEqual(result["candidate_snapshot"], "full_pool.json")
        self.assertEqual(result["confirmed_candidate_snapshot"], "confirmed.json")
        self.assertFalse(result["camera_candidate_resolution_complete"])
        self.assertFalse(result["motion_authorized"])

    def test_changed_goal_pool_or_missing_remaining_candidate_cannot_complete(self):
        changes = (
            {"expected_stand_count": 6}, {"goal_completed": False},
            {"candidate_pool_count": 5}, {"remaining_candidate_uids": []},
            {"confirmed_candidate_uids": ["unknown"] * 5},
            {"candidate_snapshot_sha256": "f" * 64},
            {"candidate_goal_progress_sha256": "not-a-hash"},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_completed_qr_goal({**fields(), **change},
                    snapshot_sha256="a" * 64, coverage=coverage())

    def test_changed_admitted_policy_rejected(self):
        value = deepcopy(coverage())
        value["inspection_pool_policy"]["inspection_pool_limit"] = 99
        with self.assertRaisesRegex(ValueError, "admitted pool policy"):
            validate_completed_qr_goal(fields(), snapshot_sha256="a" * 64, coverage=value)

    def test_mission_handoff_allows_full_bounded_pool_not_a_subset(self):
        uids = tuple(f"candidate_{i}" for i in range(6))
        snapshot = SimpleNamespace(candidates=tuple(SimpleNamespace(candidate_uid=uid) for uid in uids))
        decision = SimpleNamespace(admitted_candidate_uids=uids, expected_stand_count=5)
        _require_exact_two_snapshot_population(snapshot, decision, expected_stand_count=5)
        with self.assertRaisesRegex(RuntimeError, "differs from admission"):
            _require_exact_two_snapshot_population(
                SimpleNamespace(candidates=snapshot.candidates[:5]), decision,
                expected_stand_count=5,
            )

    def test_mission_handoff_rejects_changed_goal_or_overbudget_pool(self):
        snapshot = SimpleNamespace(candidates=tuple(
            SimpleNamespace(candidate_uid=f"candidate_{i:02d}") for i in range(11)
        ))
        decision = SimpleNamespace(admitted_candidate_uids=tuple(c.candidate_uid for c in snapshot.candidates), expected_stand_count=5)
        with self.assertRaisesRegex(RuntimeError, "exceeds_limit"):
            _require_exact_two_snapshot_population(snapshot, decision, expected_stand_count=5)
        with self.assertRaisesRegex(RuntimeError, "changed the expected QR goal"):
            _require_exact_two_snapshot_population(snapshot, decision, expected_stand_count=6)
