"""Camera-first integration through the production inspection adapter."""
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.approach import CandidateObservation, _CandidateObservationFrame
from scripts.aufgabe04.real_robot.candidate.inspection_adapters import (
    execute_local_candidate_inspection, lidar_support_goal_is_useful,
)
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import CandidateInspectionRouteUnavailableError
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from tests.aufgabe04.test_lidar_alignment_arrival import calibration
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture


class CandidateLidarHandoffTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        snapshot, self.registry, planning = hint_fixture()
        self.source = SimpleNamespace(
            snapshot=snapshot, camera_calibration=calibration(), camera_timeout_sec=90.,
            max_candidate_inspection_views=4, approach_offset_m=.55,
            camera_arrival_range_slack_m=.20,
        )
        self.initial = _CandidateObservationFrame(self.source, snapshot.candidates[0], planning, None)
        self.fresh = replace(self.initial, observation_pose=planning.current_pose)
        self.events = []
        self.recovery = Mock(side_effect=AssertionError("unexpected LiDAR recovery"))
        self.admit = Mock(return_value=self.initial)
        self.capture = Mock(side_effect=self.resolved)
        self.effects = SimpleNamespace(
            capture_lidar_view=Mock(side_effect=AssertionError("no preliminary LiDAR capture")),
            admit_planning_frame=Mock(), load_route_uncertainty_readiness=Mock(),
            run_centering_turn=None, capture_observation=self.capture,
            event_sink=lambda path, event: self.events.append(event), clock=lambda: 30.,
        )
        self.motion = Mock(side_effect=AssertionError("unexpected motion"))

    def resolved(self, request):
        return CandidateObservation(request.output_dir / "recommendation.json", "QR_A", None)

    def unavailable(self, request):
        raise CandidateObservationUnavailableError(
            candidate_uid=self.initial.candidate.candidate_uid,
            observation_attempt_index=request.attempt_index, reason="head_not_observable",
            process_evidence={"observer_started": True}, status_evidence={},
        )

    def execute(self):
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.create_lidar_camera_recovery",
                   return_value=self.recovery):
            return execute_local_candidate_inspection(
                observation_frame=self.initial, source_config=self.source, effects=self.effects,
                source_registry=self.registry, candidate_root=self.root,
                candidate_run_id="run", candidate_index=0, admit_arrival=self.admit,
                admit_planning=Mock(side_effect=AssertionError("unexpected planning")),
                move_certified_opposite=Mock(side_effect=AssertionError("unexpected opposite motion")),
                execute_motion=self.motion, frame_type=_CandidateObservationFrame,
                request_type=SimpleNamespace,
                observation_request_type=lambda candidate, output_dir, attempt_index, **kw: SimpleNamespace(
                    candidate=candidate, output_dir=output_dir, attempt_index=attempt_index, **kw),
            )

    def test_first_arrival_uses_camera_without_any_lidar_acquisition(self):
        result, frame = self.execute()
        self.assertEqual(result.qr_id, "QR_A")
        self.assertIs(frame, self.initial)
        self.admit.assert_called_once()
        self.capture.assert_called_once()
        self.recovery.assert_not_called()
        self.effects.capture_lidar_view.assert_not_called()
        self.motion.assert_not_called()
        self.assertEqual([(e["event"], e["state"]) for e in self.events], [
            ("arrival_admission", "started"), ("arrival_admission", "returned"),
            ("observer_capture", "started"), ("observer_capture", "returned"),
        ])

    def test_recorded_insufficient_fits_return_to_camera_after_each_move(self):
        order = []
        def capture(request):
            order.append(("camera", request.attempt_index))
            if request.attempt_index < 3:
                return self.unavailable(request)
            return self.resolved(request)
        self.capture.side_effect = capture
        # The recorded 2/3, 0/3, 0/3 cohorts never produce normal alignment.
        fits = iter((2, 0, 0))
        def recover(frame):
            self.assertEqual(order[-1][0], "camera")
            order.append(("lidar", next(fits)))
            return self.fresh, {"motion_completed": True, "head_alignment_verified": False,
                                "camera_centered_verified": False}, None
        self.recovery.side_effect = recover
        result, _ = self.execute()
        self.assertEqual(result.qr_id, "QR_A")
        self.assertEqual(order, [("camera", 0), ("lidar", 2), ("camera", 1),
                                 ("lidar", 0), ("camera", 2), ("lidar", 0), ("camera", 3)])
        self.assertEqual(self.admit.call_count, 4)
        self.motion.assert_not_called()  # Admission cannot add a second recovery move.
        self.assertFalse(list(self.root.glob("lidar_recovery_*/candidate_arrival_admission.json")))

    def test_verified_recovery_keeps_fitted_pose_without_old_centroid_readmission(self):
        self.capture.side_effect = lambda r: self.unavailable(r) if r.attempt_index == 0 else self.resolved(r)
        self.recovery.side_effect = None
        self.recovery.return_value = (self.fresh, {
            "motion_completed": True, "head_alignment_verified": True,
            "camera_centered_verified": True,
        }, object())
        _, frame = self.execute()
        self.assertIs(frame, self.fresh)
        self.admit.assert_called_once()  # Initial admission only.
        receipt = next(self.root.glob("lidar_recovery_*/candidate_arrival_admission.json"))
        evidence = json.loads(receipt.read_text())
        self.assertTrue(evidence["requires_live_target_association"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertEqual(evidence["admission_kind"], "fresh_lidar_calibrated_camera_alignment")

    def test_head_alignment_without_centering_still_requires_passive_admission(self):
        self.capture.side_effect = lambda r: self.unavailable(r) if r.attempt_index == 0 else self.resolved(r)
        self.recovery.side_effect = None
        self.recovery.return_value = (self.fresh, {
            "motion_completed": True, "head_alignment_verified": True,
            "camera_centered_verified": False,
        }, object())
        self.execute()
        self.assertEqual(self.admit.call_count, 2)
        self.assertFalse(list(self.root.glob("lidar_recovery_*/candidate_arrival_admission.json")))
        self.motion.assert_not_called()

    def test_post_recovery_arrival_rejection_does_not_start_another_correction_move(self):
        self.capture.side_effect = self.unavailable
        self.recovery.side_effect = None
        self.recovery.return_value = (self.fresh, {
            "motion_completed": True, "head_alignment_verified": False,
            "camera_centered_verified": False,
        }, None)
        error = CandidateObservationUnavailableError(
            candidate_uid=self.initial.candidate.candidate_uid, observation_attempt_index=1,
            reason="candidate_arrival_geometry_rejected", process_evidence={},
            status_evidence={"reasons": ["bearing_error_above_maximum"]})
        self.admit.side_effect = [self.initial, error]
        with self.assertRaises(CandidateObservationUnavailableError):
            self.execute()
        self.motion.assert_not_called()
        self.capture.assert_called_once()
        self.recovery.assert_called_once()

    def test_calibrated_first_arrival_rejection_cannot_start_map_only_correction(self):
        self.admit.side_effect = CandidateObservationUnavailableError(
            candidate_uid=self.initial.candidate.candidate_uid, observation_attempt_index=0,
            reason="candidate_arrival_geometry_rejected", process_evidence={},
            status_evidence={"reasons": ["bearing_error_above_maximum"]})
        with self.assertRaises(CandidateObservationUnavailableError):
            self.execute()
        self.capture.assert_not_called()
        self.recovery.assert_not_called()
        self.motion.assert_not_called()

    def test_interrupted_recovery_retains_handoff_failure(self):
        self.capture.side_effect = self.unavailable
        self.recovery.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.execute()
        failures = [e for e in self.events if e["event"] == "lidar_recovery" and e["state"] == "failed"]
        self.assertEqual(failures[0]["exception_type"], "KeyboardInterrupt")
        self.motion.assert_not_called()

    def test_materialized_probe_checks_usefulness_deadline_and_shared_route_budget(self):
        for kind in ("redundant", "expired", "useful", "sparse", "supported"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                self.events.clear()
                self.source.map_yaml = root / "map.yaml"
                self.source.semantic_map_id = "map"
                self.source.plan = object()
                self.source.snapshot_path = root / "snapshot.json"
                self.source.inflation_radius_m = .2
                self.source.candidate_transit_radius_m = .31
                self.source.physical_clearance = {"minimum_active_standoff_m": .33}
                goal = {"x_m": -.57, "y_m": 0., "yaw_rad": 0.} if kind == "redundant" else {
                    "x_m": -.3, "y_m": -.52, "yaw_rad": math.pi/3}
                def plan(request):
                    request.output_dir.mkdir(parents=True)
                    (request.output_dir / "pipeline_summary.json").write_text(json.dumps({
                        "selected_approach_pose": goal}))
                    return object()
                self.effects.plan_preapproach = plan
                guard = Mock()
                if kind == "expired":
                    guard.side_effect = [None, CandidateInspectionRouteUnavailableError(
                        "elapsed during planning", reason_code="lidar_recovery_time_budget_exhausted")]
                support_hint = None
                if kind in ("sparse", "supported"):
                    support_hint = {"center_odom": {"x_m": 0., "y_m": 0.},
                        "tangent_odom_rad": math.pi/3 if kind == "sparse" else -math.pi/6,
                        "angle_uncertainty_rad": .03, "angular_step_rad": .028,
                        "scan_pose_robot": {"x_m": .04, "y_m": 0., "yaw_rad": 0.},
                        "source_evidence_paths": ["recorded_view.json"], "motion_authorized": False}
                def factory(**bound):
                    def recover(frame):
                        moved = bound["plan_and_move"](
                            frame, -2*math.pi/3, root / "probe", 1, None,
                            purpose="lidar_axis_hint", offset=.6, before_motion=guard, lidar_support_hint=support_hint)
                        return moved, {"motion_completed": True, "head_alignment_verified": False,
                                       "camera_centered_verified": False}, None
                    return recover
                motion = Mock()
                with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.create_lidar_camera_recovery",
                           side_effect=factory), patch(
                        "scripts.aufgabe04.real_robot.candidate.inspection_adapters.execute_candidate_inspection",
                        side_effect=lambda **kw: kw["effects"].recover_lidar(self.initial, root / "recovery", 1)):
                    def execute():
                        return execute_local_candidate_inspection(
                            observation_frame=self.initial, source_config=self.source, effects=self.effects,
                            source_registry=self.registry, candidate_root=root, candidate_run_id="run",
                            candidate_index=0, admit_arrival=self.admit,
                            admit_planning=lambda **kw: (self.source, self.initial.candidate,
                                self.initial.planning_frame.current_pose, self.initial.planning_frame, None),
                            move_certified_opposite=Mock(), execute_motion=motion,
                            frame_type=_CandidateObservationFrame, request_type=SimpleNamespace,
                            observation_request_type=SimpleNamespace)
                    if kind in ("useful", "supported"):
                        execute()
                        motion.assert_called_once()
                    else:
                        with self.assertRaises(CandidateInspectionRouteUnavailableError):
                            execute()
                        motion.assert_not_called()
                starts = [e for e in self.events if e["event"] == "inspection_route_proposal_started"]
                self.assertEqual(len(starts), 1)
                if kind not in ("useful", "supported"):
                    rejected = [e for e in self.events if e["event"] == "inspection_route_proposal_rejected"]
                    self.assertEqual(rejected[0]["reason_code"], "lidar_support_goal_not_useful"
                                     if kind == "redundant" else "lidar_support_goal_too_sparse" if kind == "sparse"
                                     else "lidar_recovery_time_budget_exhausted")


class LidarSupportGoalUsefulnessTest(unittest.TestCase):
    def test_tiny_displacement_or_yaw_change_is_not_an_independent_view(self):
        self.assertFalse(lidar_support_goal_is_useful(start=Pose2D(.55, 0., 0.),
            goal={"x_m": .52, "y_m": 0., "yaw_rad": math.pi/2}, target_x_m=0., target_y_m=0.))

    def test_separated_view_or_meaningfully_closer_range_can_add_support(self):
        for x, y in ((.275, .55*math.sin(math.pi/3)), (.49, 0.)):
            with self.subTest(x=x, y=y):
                self.assertTrue(lidar_support_goal_is_useful(start=Pose2D(.55, 0., 0.),
                    goal={"x_m": x, "y_m": y}, target_x_m=0., target_y_m=0.))

    def test_same_axial_view_on_opposite_side_is_not_independent_support(self):
        self.assertFalse(lidar_support_goal_is_useful(start=Pose2D(.55, 0., 0.),
            goal={"x_m": -.55, "y_m": 0.}, target_x_m=0., target_y_m=0.))
