import json
import math
from pathlib import Path
import tempfile
import unittest

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
