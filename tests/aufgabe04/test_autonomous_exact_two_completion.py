from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.real_robot import autonomous_coverage_mission as mission
from scripts.aufgabe04.real_robot import autonomous_exact_two_completion as completion


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class AutonomousExactTwoCompletionTests(unittest.TestCase):
    def test_structural_completion_preserves_camera_evidence_order(self):
        events: list[str] = []
        candidate = SimpleNamespace(candidate_uid="candidate_0")
        snapshot = SimpleNamespace(candidates=(candidate,))
        camera_decision = SimpleNamespace(
            ready=True,
            reasons=(),
            expected_stand_count=1,
            active_candidate_count=1,
            selected_candidate_count=1,
            camera_seed_selection_mode="strict_exact",
            selected_candidate_uids=("candidate_0",),
            boundary_fill_candidate_uids=(),
            boundary_audit_only_candidate_uids=(),
            excluded_candidate_uids=(),
            admitted_candidate_uids=("candidate_0",),
            multi_view_candidate_uids=("candidate_0",),
            single_view_candidate_uids=(),
            lidar_static_map_admitted_candidate_uids=("candidate_0",),
            lidar_boundary_provisional_candidate_uids=(),
            lidar_population_retained_candidate_uids=("candidate_0",),
            source_registry_sha256=HASH_A,
        )
        handoff = SimpleNamespace(motion_authorized=False)

        def record(name: str, value):
            def callback(*args):
                del args
                events.append(name)
                return value

            return callback

        request = completion.ExactTwoCameraCompletionRequest(
            run_mode="execute-exact-two-camera",
            session_id="structural_test",
            expected_stand_count=1,
            survey_root=Path("survey"),
            plan=SimpleNamespace(),
            progress=SimpleNamespace(),
            registry=SimpleNamespace(),
            terminal_checkpoint=SimpleNamespace(
                manifest_path=Path("terminal_checkpoint.json"),
                manifest_sha256=HASH_A,
            ),
            terminal_checkpoint_parent=None,
            lidar_admission_path=Path("lidar_admission.json"),
            lidar_admission_sha256=HASH_B,
            lidar_decision=SimpleNamespace(),
            camera_admission_path=Path("camera_admission.json"),
            candidate_snapshot_path=Path("candidate_snapshot.json"),
            camera_handoff_path=Path("camera_handoff.json"),
            completed_coverage_legs=2,
            legs_completed_this_run=2,
            coverage_status=SimpleNamespace(
                to_summary_fields=lambda: {"coverage_complete": True}
            ),
        )
        effects = SimpleNamespace(
            evaluate_exact_two_camera_admission=record(
                "evaluate_camera_admission", camera_decision
            ),
            write_exact_two_camera_admission=record(
                "write_camera_admission", HASH_C
            ),
            exact_two_camera_admission_sha256=record(
                "hash_camera_admission", HASH_C
            ),
            build_exact_two_camera_snapshot=record("build_snapshot", snapshot),
            write_snapshot=record("write_snapshot", HASH_A),
            snapshot_sha256=record("hash_snapshot", HASH_A),
            create_exact_two_camera_handoff=record("create_handoff", handoff),
            write_exact_two_camera_handoff=record("write_handoff", HASH_B),
            exact_two_camera_handoff_sha256=record("hash_handoff", HASH_B),
        )

        outcome = completion.complete_exact_two_camera(request, effects)

        self.assertIsInstance(outcome, completion.CoverageExactTwoCameraReady)
        self.assertEqual(
            events,
            [
                "evaluate_camera_admission",
                "write_camera_admission",
                "hash_camera_admission",
                "build_snapshot",
                "write_snapshot",
                "hash_snapshot",
                "create_handoff",
                "write_handoff",
                "hash_handoff",
            ],
        )
        self.assertFalse(outcome.motion_authorized)
        self.assertTrue(outcome.to_mission_summary()["candidate_snapshot_ready"])

    def test_mission_reexports_exact_two_public_contract(self):
        self.assertIs(
            mission.CoverageExactTwoCameraAdmissionError,
            completion.CoverageExactTwoCameraAdmissionError,
        )
        self.assertIs(
            mission.CoverageExactTwoCameraHandoffRequest,
            completion.CoverageExactTwoCameraHandoffRequest,
        )
        self.assertIs(
            mission.CoverageExactTwoCameraReady,
            completion.CoverageExactTwoCameraReady,
        )
        self.assertEqual(
            mission.COVERAGE_EXACT_TWO_CAMERA_READY,
            completion.COVERAGE_EXACT_TWO_CAMERA_READY,
        )

    def test_sibling_does_not_import_parent_and_parent_uses_public_api(self):
        sibling_path = Path(completion.__file__)
        sibling_tree = ast.parse(
            sibling_path.read_text(encoding="utf-8"),
            filename=str(sibling_path),
        )
        imported_modules = set()
        for node in ast.walk(sibling_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        self.assertNotIn(
            "scripts.aufgabe04.real_robot.autonomous_coverage_mission",
            imported_modules,
        )
        self.assertTrue(
            {"rclpy", "subprocess", "geometry_msgs", "nav_msgs"}.isdisjoint(
                imported_modules
            )
        )
        sibling_source = sibling_path.read_text(encoding="utf-8")
        self.assertNotIn("input(", sibling_source)
        self.assertNotIn("/cmd_vel", sibling_source)

        parent_path = Path(mission.__file__)
        parent_tree = ast.parse(
            parent_path.read_text(encoding="utf-8"),
            filename=str(parent_path),
        )
        imports = [
            node
            for node in ast.walk(parent_tree)
            if isinstance(node, ast.ImportFrom)
            and node.module
            == "scripts.aufgabe04.real_robot.autonomous_exact_two_completion"
        ]
        self.assertEqual(len(imports), 1)
        self.assertFalse(
            any(alias.name.startswith("_") for alias in imports[0].names)
        )
        self.assertTrue(
            {alias.name for alias in imports[0].names}.issubset(
                set(completion.__all__)
            )
        )
        parent_definitions = {
            node.name
            for node in parent_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertTrue(
            {
                "CoverageExactTwoCameraAdmissionError",
                "CoverageExactTwoCameraHandoffRequest",
                "CoverageExactTwoCameraReady",
            }.isdisjoint(parent_definitions)
        )


if __name__ == "__main__":
    unittest.main()
