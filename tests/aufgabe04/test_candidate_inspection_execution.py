import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, CandidateInspectionRouteUnavailableError,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.inspection_policy import (
    CandidateInspectionState, candidate_view_options, validate_inspection_budget,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError


class CandidateInspectionExecutionTest(unittest.TestCase):
    def test_joint_recommendation_at_first_view_never_requests_another_local_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = CandidateObservation(root / "recommendation.json", "QR_003", None)
            captures = []

            def capture(frame, output, index):
                captures.append(index)
                return observation

            result, frame = execute_candidate_inspection(
                candidate_uid="candidate", candidate_root=root, initial_frame=0., max_views=8,
                effects=CandidateInspectionEffects(
                    capture=capture, canonical_normal=lambda frame: frame,
                    move_view=lambda *args: self.fail("resolved first view requested another local move"),
                    move_opposite=lambda *args: self.fail("resolved front requested an opposite view"),
                    progress_evidence=lambda *args: self.fail("resolved view became advisory"),
                ),
            )
            self.assertIs(result, observation)
            self.assertEqual(frame, 0.)
            self.assertEqual(captures, [0])
            progress = json.loads((root / "inspection_progress.json").read_text())
            self.assertTrue(progress["joint_observation_ready"])
            self.assertEqual(progress["local_view_count"], 1)

    def test_oblique_decoded_qr_is_preserved_until_new_view_has_joint_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures, moves = [], []

            def capture(frame, output, index):
                captures.append(index)
                if index == 0:
                    return CandidateObservation(None, None, None, root / "advisory.json")
                return CandidateObservation(output / "recommendation.json", "QR_001", None)

            def move(frame, normal, output, index, source):
                moves.append(normal)
                return normal

            result, final_frame = execute_candidate_inspection(
                candidate_uid="candidate", candidate_root=root, initial_frame=0.0, max_views=8,
                effects=CandidateInspectionEffects(
                    capture=capture, canonical_normal=lambda frame: frame, move_view=move,
                    move_opposite=lambda *args: self.fail("advisory was treated as backside"),
                    progress_evidence=lambda *args: {"classification": "oblique", "qr_id": "QR_001",
                                                     "camera_relative_yaw_rad": math.radians(51)},
                ),
            )
            self.assertEqual(captures, [0, 1])
            self.assertAlmostEqual(moves[0], math.radians(51))
            self.assertEqual(result.qr_id, "QR_001")
            progress = json.loads((root / "inspection_progress.json").read_text())
            self.assertEqual(progress["provisional_qr_ids"], ["QR_001"])
            self.assertEqual(len(progress["view_history"]), 2)
            self.assertGreater(len(list((root / "inspection_history").glob("*.json"))), 1)
            self.assertTrue(progress["joint_observation_ready"])

    def test_certified_backside_moves_opposite_then_observes_same_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moved = []
            def opposite(frame, observation, output, index):
                moved.append(index)
                return math.pi
            result, frame = execute_candidate_inspection(
                candidate_uid="candidate", candidate_root=root, initial_frame=0.0, max_views=2,
                effects=CandidateInspectionEffects(
                    capture=lambda frame, output, index: (
                        CandidateObservation(None, None, root / "axis.json") if index == 0
                        else CandidateObservation(output / "recommendation.json", "QR_A", None)),
                    canonical_normal=lambda frame: frame,
                    move_view=lambda *args: self.fail("valid backside skipped opposite view"),
                    move_opposite=opposite, progress_evidence=lambda *args: {},
                ),
            )
            self.assertEqual(moved, [1])
            self.assertEqual(result.qr_id, "QR_A")

    def test_unavailable_views_exhaust_locally_without_repeating_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normals, captures = [], []
            def capture(frame, output, index):
                captures.append(index)
                raise CandidateObservationUnavailableError(
                    candidate_uid="candidate", observation_attempt_index=index,
                    reason="unobservable", process_evidence={}, status_evidence={},
                )
            def move(frame, normal, output, index, source):
                normals.append(normal)
                return normal
            with self.assertRaises(CandidateObservationUnavailableError) as raised:
                execute_candidate_inspection(
                    candidate_uid="candidate", candidate_root=root, initial_frame=0.0, max_views=4,
                    effects=CandidateInspectionEffects(
                        capture=capture, canonical_normal=lambda frame: frame, move_view=move,
                        move_opposite=lambda *args: None, progress_evidence=lambda *args: {},
                    ),
                )
            self.assertEqual(captures, [0, 1, 2, 3])
            self.assertEqual(len(normals), 3)
            self.assertEqual(len(set(normals)), 3)
            self.assertEqual(raised.exception.reason, "candidate_local_inspection_exhausted")

    def test_one_view_budget_does_not_move_on_axis_only_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(CandidateObservationUnavailableError):
                execute_candidate_inspection(
                    candidate_uid="candidate", candidate_root=root, initial_frame=0.0, max_views=1,
                    effects=CandidateInspectionEffects(
                        capture=lambda *args: CandidateObservation(None, None, root / "axis.json"),
                        canonical_normal=lambda frame: frame,
                        move_view=lambda *args: self.fail("budget exceeded"),
                        move_opposite=lambda *args: self.fail("budget exceeded"),
                        progress_evidence=lambda *args: {},
                    ),
                )

    def test_third_view_terminal_failure_is_recorded_without_another_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures, moves = [], []
            terminal = RuntimeError("observer terminal failure: " + "x" * 2048)

            def capture(frame, output, index):
                captures.append(index)
                if index == 2:
                    raise terminal
                return CandidateObservation(None, None, None, output / "inspection.json")

            def move(frame, normal, output, index, source):
                moves.append(index)
                return normal

            with self.assertRaises(RuntimeError) as caught:
                execute_candidate_inspection(
                    candidate_uid="candidate", candidate_root=root, initial_frame=0., max_views=8,
                    effects=CandidateInspectionEffects(
                        capture=capture, canonical_normal=lambda frame: frame, move_view=move,
                        move_opposite=lambda *args: self.fail("terminal failure authorized opposite motion"),
                        progress_evidence=lambda *args: {"classification": "unobservable"},
                    ),
                )

            self.assertIs(caught.exception, terminal)
            self.assertNotIsInstance(caught.exception, CandidateObservationUnavailableError)
            self.assertEqual(captures, [0, 1, 2])
            self.assertEqual(moves, [1, 2])
            progress = json.loads((root / "inspection_progress.json").read_text())
            self.assertEqual(progress["local_view_count"], 3)
            self.assertEqual(progress["termination_reason"], "observer_terminal_failure")
            self.assertFalse(progress["view_budget_exhausted"])
            self.assertFalse(progress["joint_observation_ready"])
            self.assertFalse(progress["motion_authorized"])
            failure = progress["view_history"][-1]
            self.assertEqual(failure["outcome"], "observation_terminal_failure")
            self.assertEqual(len(failure["reason"]), 1024)
            self.assertEqual(failure["observation"]["exception_type"], "RuntimeError")
            self.assertEqual(failure["observation"]["observer_attempt_index"], 2)
            self.assertEqual(failure["observation"]["observer_output_dir"],
                             str(root / "camera_lidar_attempt_02"))
            self.assertFalse(failure["observation"]["completion_authorized"])
            receipt = load_content_hashed_json(
                Path(progress["latest_revision_path"]),
                hash_field="candidate_inspection_progress_sha256",
            )
            self.assertEqual(receipt["view_history"], progress["view_history"])

    def test_terminal_identity_conflict_records_attempt_but_no_joint_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "QR identity conflict"):
                execute_candidate_inspection(
                    candidate_uid="candidate", candidate_root=root, initial_frame=0., max_views=8,
                    effects=CandidateInspectionEffects(
                        capture=lambda frame, output, index: (
                            CandidateObservation(None, None, None, output / "inspection.json") if index == 0
                            else CandidateObservation(output / "recommendation.json", "QR_B", None)
                        ),
                        canonical_normal=lambda frame: frame,
                        move_view=lambda frame, normal, output, index, source: normal,
                        move_opposite=lambda *args: self.fail("identity conflict authorized opposite motion"),
                        progress_evidence=lambda *args: {"classification": "front_readable", "qr_id": "QR_A"},
                    ),
                )
            progress = json.loads((root / "inspection_progress.json").read_text())
            self.assertEqual(progress["local_view_count"], 2)
            self.assertEqual(progress["provisional_qr_ids"], ["QR_A"])
            self.assertEqual(progress["termination_reason"], "observer_terminal_failure")
            self.assertFalse(progress["joint_observation_ready"])

    def test_interrupted_capture_is_recorded_and_original_interrupt_propagates(self):
        for terminal in (KeyboardInterrupt(), SystemExit(130)):
            with self.subTest(error_type=type(terminal).__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)

                def capture(*args):
                    raise terminal

                with self.assertRaises(type(terminal)) as caught:
                    execute_candidate_inspection(
                        candidate_uid="candidate", candidate_root=root, initial_frame=0., max_views=8,
                        effects=CandidateInspectionEffects(
                            capture=capture, canonical_normal=lambda frame: frame,
                            move_view=lambda *args: self.fail("interruption resumed motion"),
                            move_opposite=lambda *args: self.fail("interruption resumed motion"),
                            progress_evidence=lambda *args: {},
                        ),
                    )
                self.assertIs(caught.exception, terminal)
                progress = json.loads((root / "inspection_progress.json").read_text())
                self.assertEqual(progress["local_view_count"], 1)
                self.assertEqual(progress["termination_reason"], "observer_terminal_failure")

    def test_diagnostic_write_failure_cannot_mask_original_terminal_error(self):
        terminal = RuntimeError("observer failed")
        persistence_error = OSError("diagnostic disk unavailable")
        with tempfile.TemporaryDirectory() as directory:
            def capture(*args):
                raise terminal

            with patch(
                "scripts.aufgabe04.real_robot.candidate.inspection_execution.write_content_hashed_json",
                side_effect=persistence_error,
            ), self.assertRaises(RuntimeError) as caught:
                execute_candidate_inspection(
                    candidate_uid="candidate", candidate_root=Path(directory), initial_frame=0., max_views=8,
                    effects=CandidateInspectionEffects(
                        capture=capture, canonical_normal=lambda frame: frame,
                        move_view=lambda *args: self.fail("failed diagnostics resumed motion"),
                        move_opposite=lambda *args: self.fail("failed diagnostics resumed motion"),
                        progress_evidence=lambda *args: {},
                    ),
                )
            self.assertIs(caught.exception, terminal)
            self.assertIs(caught.exception.__cause__, persistence_error)

    def test_qr_conflict_cannot_be_replaced_by_later_joint_observation(self):
        state = CandidateInspectionState("candidate", 4)
        state.record(outcome="inspection_pending", normal=0.0, observation={"qr_id": "QR_A"})
        with self.assertRaisesRegex(ValueError, "QR identity conflict"):
            state.record(outcome="resolved", normal=1.0, observation={"qr_id": "QR_B"})
        self.assertFalse(state.to_dict()["joint_observation_ready"])

    def test_advisory_sign_is_two_search_hypotheses_not_directed_axis(self):
        options = candidate_view_options(0.0, classification="front_readable",
                                         achieved_normals=[0.0], attempted_normals=[],
                                         advisory_yaw_rad=math.radians(-51))
        self.assertAlmostEqual(options[0], math.radians(51))
        self.assertAlmostEqual(options[1], math.radians(-51))
        self.assertTrue(all(abs(value) >= math.radians(20) for value in options))

    def test_budget_requires_bounded_actual_integer(self):
        for value in (True, 0, 17, 2.0, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_inspection_budget(value)


if __name__ == "__main__":
    unittest.main()
