import json
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation
from scripts.aufgabe04.real_robot.candidate.camera_distance_recovery import (
    distance_recovery_goal_is_useful, select_camera_distance_recovery,
)
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects, CandidateInspectionRouteUnavailableError,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.observer.camera_framing import (
    build_camera_framing_hint, validate_camera_framing_hint,
)
from scripts.aufgabe04.real_robot.observer.diagnostics import load_passive_observer_status
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.candidate_preapproach_models import CandidatePreapproachUnreachableError
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, execute_candidate_approach_phase,
)
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_detected_station_exploration import write_free_map


def framing_hint(**changes):
    fields = dict(
        target_key="candidate", source_image_stamp_sec=123.0,
        front_evidence_verified=True, candidate_associated=True,
        reason="head_qr_geometry_mismatch", range_m=.413, optical_depth_m=.372,
        image_size=(640, 480), head_bounds=(443., 160., 580., 294.),
        fx_px=520., fy_px=520.,
    )
    return build_camera_framing_hint(**{**fields, **changes})


class CameraFramingTest(unittest.TestCase):
    def test_complete_off_center_head_does_not_itself_require_distance_change(self):
        self.assertIsNone(framing_hint(reason="nominal_roi_clipped"))
        hint = framing_hint()
        self.assertLess(hint["minimum_range_m"], .413)
        self.assertGreater(hint["maximum_range_m"], .7)
        self.assertFalse(hint["motion_authorized"])

    def test_unverified_unassociated_or_malformed_image_evidence_cannot_hint(self):
        for values in (
            {"front_evidence_verified": False}, {"candidate_associated": False},
            {"head_bounds": (580., 160., 650., 294.)}, {"fx_px": 0.},
            {"source_image_stamp_sec": float("nan")}, {"image_size": (True, 480)},
        ):
            with self.subTest(values=values):
                self.assertIsNone(framing_hint(**values))

    def test_frustum_envelope_scales_with_observed_size_and_optical_depth(self):
        regular = framing_hint()
        distant = framing_hint(range_m=.785, optical_depth_m=.744)
        self.assertAlmostEqual(distant["minimum_range_m"] - .041,
                               2 * (regular["minimum_range_m"] - .041))
        self.assertAlmostEqual(distant["maximum_range_m"] - .041,
                               2 * (regular["maximum_range_m"] - .041))

    def test_latest_status_carries_only_valid_motion_neutral_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            hint = framing_hint()
            path.write_text(json.dumps({"state": "obsolete_detector_result", "camera_framing": hint}))
            self.assertEqual(load_passive_observer_status(path).to_dict()["camera_framing"], hint)
            for mutation in ({"front_evidence_verified": False}, {"motion_authorized": True},
                             {"range_m": True}, {"minimum_range_m": 100}):
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps({"state": "obsolete_detector_result",
                                                "camera_framing": {**hint, **mutation}}))
                    status = load_passive_observer_status(path)
                    self.assertIsNone(status.camera_framing)
                    self.assertIsNone(status.load_error, "advisory data must not change timeout classification")


class CameraDistancePolicyTest(unittest.TestCase):
    def select(self, hint=None, **changes):
        fields = dict(candidate_uid="candidate", current_range_m=.413,
                      preferred_range_m=.7, maximum_allowed_range_m=.9)
        return select_camera_distance_recovery(framing_hint() if hint is None else hint,
                                               **{**fields, **changes})

    def test_outward_proposals_stay_bounded_and_never_shrink_to_collision_floor(self):
        recovery = self.select()
        self.assertEqual(len(recovery.standoffs_m), 3)
        self.assertAlmostEqual(recovery.standoffs_m[0], .7)
        self.assertTrue(all(offset >= .513 for offset in recovery.standoffs_m))
        self.assertFalse(recovery.to_dict()["motion_authorized"])

    def test_no_global_minimum_distance_or_unbound_target_correction(self):
        self.assertIsNone(self.select(candidate_uid="other"))
        self.assertIsNone(self.select(current_range_m=.7))
        self.assertIsNone(self.select(preferred_range_m=.50))
        self.assertIsNone(self.select(hint={}))
        self.assertIsNone(self.select(hint={**framing_hint(), "front_evidence_verified": False}))

    def test_camera_range_envelope_intersects_existing_arrival_limit(self):
        self.assertIsNone(self.select(hint={**framing_hint(), "maximum_range_m": .49}))
        recovery = self.select(maximum_allowed_range_m=.59)
        self.assertEqual(recovery.standoffs_m[0], .59)
        self.assertLessEqual(recovery.maximum_range_m, .59)

    def test_identical_bearing_allowed_but_quantized_inward_or_orbit_is_not(self):
        fields = dict(start_range_m=.413, goal_range_m=.68, requested_normal_rad=0.,
                      achieved_normal_rad=0., minimum_range_m=.2, maximum_range_m=.9)
        self.assertTrue(distance_recovery_goal_is_useful(**fields))
        for changes in ({"goal_range_m": .45}, {"achieved_normal_rad": math.radians(6)},
                        {"goal_range_m": .91}, {"goal_range_m": float("nan")}):
            with self.subTest(changes=changes):
                self.assertFalse(distance_recovery_goal_is_useful(**{**fields, **changes}))


class CameraDistanceExecutionTest(unittest.TestCase):
    def execute(self, *, max_views=3, blocked=False, proposal_budget=False, resolved=False):
        self.captures, self.recoveries, self.moves = [], [], []
        self.selection_calls = []
        def capture(frame, root, index):
            self.captures.append(index)
            if resolved and index > 0:
                return CandidateObservation(root / "recommendation.json", "QR_A", None)
            raise CandidateObservationUnavailableError(
                candidate_uid="candidate", observation_attempt_index=index, reason="unresolved",
                process_evidence={}, status_evidence={"camera_framing": framing_hint()},
            )
        def recovery(frame, evidence):
            self.selection_calls.append(frame)
            return evidence.get("camera_framing")
        def move_recovery(frame, hint, root, index, source):
            self.recoveries.append(index)
            if blocked or proposal_budget:
                raise CandidateInspectionRouteUnavailableError(
                    "occupied outward poses", reason_code=("route_proposal_budget_exhausted"
                                                            if proposal_budget else "standoff_proposals_exhausted"))
            return frame
        def move(frame, normal, root, index, source):
            self.moves.append(normal)
            return normal
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        return execute_candidate_inspection(
            candidate_uid="candidate", candidate_root=self.root, initial_frame=0., max_views=max_views,
            effects=CandidateInspectionEffects(
                capture=capture, canonical_normal=lambda frame: frame, move_view=move,
                move_opposite=lambda *args: self.fail("front hint became certified backside"),
                progress_evidence=lambda *args: {}, distance_recovery=recovery,
                move_distance_recovery=move_recovery,
            ),
        )

    def test_same_bearing_recovery_consumes_next_normal_observation_slot(self):
        result, _ = self.execute(resolved=True)
        self.assertEqual(result.qr_id, "QR_A")
        self.assertEqual(self.captures, [0, 1])
        self.assertEqual(self.recoveries, [1])
        self.assertEqual(self.moves, [])
        progress = json.loads((self.root / "inspection_progress.json").read_text())
        self.assertEqual(progress["achieved_view_normals_rad"], [0., 0.])

    def test_only_one_recovery_even_if_every_later_view_repeats_hint(self):
        with self.assertRaises(CandidateObservationUnavailableError) as caught:
            self.execute()
        self.assertEqual(self.captures, [0, 1, 2])
        self.assertEqual(self.recoveries, [1])
        self.assertEqual(len(self.selection_calls), 1)
        self.assertEqual(caught.exception.status_evidence["termination_reason"], "view_budget_exhausted")

    def test_occupied_backout_rejects_once_then_ordinary_diverse_view_remains(self):
        with self.assertRaises(CandidateObservationUnavailableError):
            self.execute(blocked=True)
        self.assertEqual(self.recoveries, [1])
        self.assertEqual(len(self.moves), 2)
        self.assertGreaterEqual(abs(self.moves[0]), math.radians(20))

    def test_recovery_cannot_bypass_camera_or_shared_route_budget(self):
        with self.assertRaises(CandidateObservationUnavailableError):
            self.execute(max_views=1)
        self.assertEqual(self.recoveries, [])
        with self.assertRaises(CandidateObservationUnavailableError) as caught:
            self.execute(proposal_budget=True)
        self.assertEqual(self.captures, [0])
        self.assertEqual(self.moves, [])
        self.assertEqual(caught.exception.status_evidence["termination_reason"], "route_proposal_budget_exhausted")


class CameraDistanceAdapterTest(unittest.TestCase):
    def run_adapter(self, *, blocked=False, bad_goal=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fixture = fixtures.AutonomousCandidateApproachTest()
        config = replace(fixture._config(root, (fixture._candidate("candidate", 0., 0.),)),
                         max_candidate_inspection_views=2)
        write_free_map(root, resolution=.05)
        current = Pose2D(.413, 0., math.pi)
        goals, proposals, motions = {}, [], []
        def plan(request):
            route_path = request.output_dir / "route.csv"
            if request.inspection_view_path is not None:
                view = json.loads(request.inspection_view_path.read_text())
                proposals.append((request, view))
                if view["purpose"] == "camera_distance_recovery" and blocked:
                    raise CandidatePreapproachUnreachableError("candidate", "occupied outward goal")
                normal = view["view_normal_rad"]
                offset = .42 if bad_goal and view["purpose"] == "camera_distance_recovery" else request.approach_offset_m
                goal = Pose2D(offset * math.cos(normal), offset * math.sin(normal),
                              math.remainder(normal + math.pi, 2 * math.pi))
                goals[str(route_path)] = goal
                request.output_dir.mkdir(parents=True, exist_ok=True)
                (request.output_dir / "pipeline_summary.json").write_text(json.dumps({
                    "selected_approach_pose": {"x_m": goal.x_m, "y_m": goal.y_m, "yaw_rad": goal.yaw_rad},
                }))
            return {"route_csv": str(route_path)}
        def run(request):
            nonlocal current
            motions.append(request)
            current = goals.get(request.sealed["route_csv"], current)
            return fixture._completed(request)
        def capture(request):
            if request.attempt_index == 0:
                raise CandidateObservationUnavailableError(
                    candidate_uid="candidate", observation_attempt_index=0, reason="geometry unresolved",
                    process_evidence={}, status_evidence={"camera_framing": framing_hint()},
                )
            return CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)
        result = execute_candidate_approach_phase(config, CandidateApproachEffects(
            select_initial_preapproach=fixture._nearest_selection, read_current_pose=lambda: current,
            plan_preapproach=plan, run_motion_leg=run, capture_observation=capture,
            validate_facing=lambda request: {"candidate_uid": "candidate"}, commit_decision=lambda request: None,
        ))
        ledger = [json.loads(line) for line in next(config.session_root.glob(
            "candidates/*/inspection_route_proposals.jsonl")).read_text().splitlines()]
        return result, proposals, motions, ledger

    def test_adapter_retains_bearing_full_snapshot_and_fresh_child_contract(self):
        result, proposals, motions, ledger = self.run_adapter()
        self.assertEqual(result.stand_count, 1)
        self.assertEqual(len(proposals), 1)
        request, view = proposals[0]
        self.assertEqual(view["purpose"], "camera_distance_recovery")
        self.assertAlmostEqual(view["view_normal_rad"], 0.)
        self.assertEqual(request.approach_offset_m, .7)
        self.assertEqual(len(request.snapshot.candidates), 1)
        self.assertEqual(len(motions), 2)
        self.assertEqual(ledger[-1]["outcome"], "route_completed")
        self.assertTrue(ledger[-1]["route_limits_unchanged"])

    def test_occupied_outward_goals_do_not_publish_and_do_not_shrink_inward(self):
        _, proposals, motions, ledger = self.run_adapter(blocked=True)
        recovery = [(request, view) for request, view in proposals if view["purpose"] == "camera_distance_recovery"]
        self.assertEqual(len(recovery), 3)
        self.assertTrue(all(request.approach_offset_m >= .513 for request, _ in recovery))
        self.assertEqual(len(motions), 2, "only initial and later diverse route may execute")
        self.assertEqual(proposals[-1][1]["purpose"], "diverse_inspection")
        self.assertEqual(sum(item["outcome"] == "proposal_rejected" for item in ledger), 3)

    def test_raster_goal_that_stays_too_close_is_rejected_before_child_motion(self):
        _, proposals, motions, ledger = self.run_adapter(bad_goal=True)
        self.assertEqual(len(motions), 2)
        reasons = [item.get("reason_code") for item in ledger if item["outcome"] == "proposal_rejected"]
        self.assertEqual(reasons, ["camera_distance_goal_not_useful"] * 3)
        self.assertEqual(proposals[-1][1]["purpose"], "diverse_inspection")


if __name__ == "__main__":
    unittest.main()
