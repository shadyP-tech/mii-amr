import copy
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.artifacts.content_store import load_content_hashed_json
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.candidate.approach import _CandidateObservationFrame
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import CandidateInspectionRouteUnavailableError
from scripts.aufgabe04.real_robot.candidate.inspection_adapters import execute_local_candidate_inspection
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition import (
    prepare_lidar_camera_arrival, run_bounded_lidar_acquisition,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import CandidateLidarView
from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError
from tests.aufgabe04.test_lidar_alignment_arrival import calibration
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture, line_receipts


def frame_fixture():
    snapshot, registry, planning = hint_fixture()
    return _CandidateObservationFrame(candidate=snapshot.candidates[0], config=SimpleNamespace(snapshot=snapshot),
        planning_frame=planning, decision_binding=SimpleNamespace(
            projection_path=Path("candidate_frame_projection.json"), projection_sha256="c"*64)), registry


class BoundedLidarAcquisitionTest(unittest.TestCase):
    def run_controller(self, *, hints=None, current_fit=None, verified_at=None,
                       unavailable=False, motion_error=None, centered=True):
        frame, _ = frame_fixture()
        calls, revisions = [], []
        hint_values = hints or [None]

        def observe(current, serial):
            index = len([c for c in calls if c[0] == "observe"])
            calls.append(("observe", serial))
            verified = verified_at == "always" or index == verified_at
            return current, hint_values[min(index, len(hint_values)-1)], current_fit, {
                "head_alignment_verified": verified,
                "camera_centered_verified": centered and verified,
                "acquisition_unavailable": unavailable,
            }

        def move(kind, current, target, serial):
            calls.append((kind, serial, target))
            if motion_error is not None:
                raise motion_error
            return current

        output = run_bounded_lidar_acquisition(initial_frame=frame, observe=observe,
            move_probe=lambda *args: move("probe", *args),
            move_aligned=lambda *args: move("aligned", *args),
            persist=lambda report: revisions.append(copy.deepcopy(report)))
        return output, calls, revisions

    def test_missing_geometry_stops_after_three_probe_moves(self):
        (_, report), calls, _ = self.run_controller()
        self.assertEqual([c[1] for c in calls if c[0] == "probe"], [1, 2, 3])
        self.assertEqual(len([c for c in calls if c[0] == "observe"]), 4)
        self.assertEqual(report["reason"], "independent_geometry_support_unavailable")
        self.assertFalse(report["head_alignment_verified"])

    def test_no_motion_probe_rejections_consume_the_three_proposal_budget(self):
        error = CandidateInspectionRouteUnavailableError("blocked", reason_code="static_clearance")
        (_, report), calls, revisions = self.run_controller(motion_error=error)
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 3)
        self.assertEqual(report["probe_moves_attempted"], 3)
        self.assertEqual(len([e for e in report["history"] if e["event"] == "no_motion_route_unavailable"]), 3)
        # Consumed attempt is durable before invoking the potentially failing effect.
        proposals = [r for r in revisions if r["history"][-1]["event"] == "motion_proposal"]
        self.assertEqual([r["probe_moves_attempted"] for r in proposals], [1, 2, 3])

    def test_two_alignment_attempt_cap_includes_no_motion_rejections(self):
        for error in (None, CandidateInspectionRouteUnavailableError("blocked")):
            with self.subTest(error=error):
                (_, report), calls, _ = self.run_controller(hints=[object()], motion_error=error)
                self.assertEqual([c[1] for c in calls if c[0] == "aligned"], [1, 2])
                self.assertEqual(report["alignment_moves_attempted"], 2)
                self.assertEqual(report["reason"], "alignment_correction_budget_exhausted")

    def test_alternating_fit_availability_does_not_reset_either_budget(self):
        (_, report), calls, _ = self.run_controller(hints=[None, object(), None, object(), None, None])
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 3)
        self.assertEqual(len([c for c in calls if c[0] == "aligned"]), 2)
        self.assertEqual(report["probe_moves_attempted"], 3)
        self.assertEqual(report["alignment_moves_attempted"], 2)

    def test_verified_arrival_exits_before_any_motion(self):
        (_, report), calls, _ = self.run_controller(hints=[object()], verified_at=0)
        self.assertTrue(report["head_alignment_verified"])
        self.assertEqual(calls, [("observe", 0)])
        self.assertFalse(report["motion_authorized"])

    def test_verified_correction_exits_after_one_attempt(self):
        (_, report), calls, _ = self.run_controller(hints=[object()], verified_at=1)
        self.assertTrue(report["head_alignment_verified"])
        self.assertEqual(len([c for c in calls if c[0] == "aligned"]), 1)

    def test_normal_aligned_but_offcenter_arrival_still_attempts_correction(self):
        (_, report), calls, _ = self.run_controller(hints=[object()], verified_at="always", centered=False)
        self.assertEqual(len([c for c in calls if c[0] == "aligned"]), 2)
        self.assertTrue(report["head_alignment_verified"])
        self.assertFalse(report["camera_centered_verified"])
        self.assertEqual(report["reason"], "alignment_correction_budget_exhausted")

    def test_unavailable_capture_never_sends_motion(self):
        (_, report), calls, _ = self.run_controller(unavailable=True, hints=[object()])
        self.assertEqual(calls, [("observe", 0)])
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(report["alignment_moves_attempted"], 0)

    def test_single_view_fit_is_a_probe_hint_and_cannot_verify_alignment(self):
        fit = SimpleNamespace(normals=lambda *_: (0., math.pi))
        (_, report), calls, _ = self.run_controller(current_fit=fit)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 3)
        self.assertFalse(any(c[0] == "aligned" for c in calls))

    def test_execution_errors_and_interruptions_propagate_without_retry(self):
        for hints in ([None], [object()]):
            for error in (RuntimeError("motion outcome unknown"), KeyboardInterrupt()):
                with self.subTest(hints=hints, error=type(error)):
                    with self.assertRaises(type(error)):
                        self.run_controller(hints=hints, motion_error=error)


class CandidateLidarAcquisitionAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame, self.registry = frame_fixture()
        self.source = SimpleNamespace(
            survey_root=self.root / "survey", plan=SimpleNamespace(survey_id="survey"),
            snapshot=self.frame.config.snapshot,
            camera_calibration=replace(calibration(), base_frame="base_footprint"),
            measured_stand_model=load_measured_physical_stand_model(Path(__file__).resolve().parents[2] /
                "configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json"),
            lidar_scan_frame="base_scan", lidar_scan_topic="/scan",
            physical_clearance={"minimum_active_standoff_m": .33}, approach_offset_m=.55,
            camera_arrival_range_slack_m=.20, camera_arrival_max_bearing_error_rad=math.radians(3.),
        )
        self.move = Mock(side_effect=AssertionError("verified/unavailable arrival must not move"))
        self.fresh = Mock(return_value=self.frame)
        self.uncertainty = SimpleNamespace(covariance=PlanarCovariance(1e-6, 0., 1e-6),
            admission_config=SimpleNamespace(localization_sigma_multiplier=1., heading_sigma_rad=math.radians(.1)))
        self.effects = SimpleNamespace(clock=Mock(side_effect=[30., 30.35]),
                                       capture_lidar_view=self.capture)

    def capture(self, request):
        start = request.observation_not_before_sec
        receipts = tuple(replace(r, receipt_id=f"arrival_{i}", viewpoint_id=request.viewpoint_id,
                                 scan_stamp_sec=start+.05+i*.08, pose_stamp_sec=start+.05+i*.08,
                                 observer_clock_sec=start+.06+i*.08)
                         for i,r in enumerate(line_receipts(increment=.01)[:3]))
        return CandidateLidarView(request.candidate_uid, request.candidate_snapshot_sha256,
            request.viewpoint_id, receipts, receipts[-1].frame_provenance.canonical_scan_pose_odom,
            receipts[-1].scan_stamp_sec, start+.3, request.output_dir / "candidate_lidar_view.json", "a"*64,
            request.output_dir / "cohort.json", "b"*64,
            tuple({"stamp_sec": r.scan_stamp_sec, "ground_frame": "base_footprint",
                   "scan_height_above_ground_m": .182, "scan_vertical_direction_x": 0.,
                   "scan_vertical_direction_y": 0., "scan_vertical_direction_z": 1.,
                   "exact_transform_stamp_sec": r.scan_stamp_sec} for r in receipts))

    def run_adapter(self, *, survey=None):
        survey = (line_receipts(increment=.01) + line_receipts(2, increment=.01)) if survey is None else survey
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.load_camera_lidar_receipts",
                   return_value=(survey, {})):
            return prepare_lidar_camera_arrival(initial_frame=self.frame, source_config=self.source,
                source_registry=self.registry, effects=self.effects, candidate_root=self.root,
                fresh_frame=self.fresh, plan_and_move=self.move,
                load_uncertainty=lambda _: self.uncertainty)

    def test_production_adapter_verifies_real_fits_and_persists_hashed_report(self):
        returned, report, hint = self.run_adapter()
        self.assertIsNotNone(hint)
        self.assertTrue(report["head_alignment_verified"], report)
        self.assertTrue(report["camera_centered_verified"])
        self.move.assert_not_called()
        saved = load_content_hashed_json(self.root / "lidar_head_acquisition/arrival_review.json",
                                        hash_field="lidar_alignment_arrival_sha256")
        self.assertEqual(saved["probe_moves_attempted"], 0)
        current = saved["history"][-1]
        self.assertEqual(current["reason"], "fresh_lidar_camera_alignment_verified")
        self.assertIn("camera_calibration_sha256", current)
        self.assertEqual(len(current["source_receipt_sha256s"]), 3)
        self.assertEqual(returned.observation_pose, returned.planning_frame.current_pose)

    def test_current_only_geometry_stays_unverified_and_failed_routes_are_bounded(self):
        self.effects.clock = Mock(side_effect=[30., 31., 32., 33.])
        self.move.side_effect = CandidateInspectionRouteUnavailableError("unsafe static route")
        _, report, hint = self.run_adapter(survey=())
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(report["probe_moves_attempted"], 3)
        # Each of three directions has exactly three bounded standoff proposals.
        self.assertEqual(self.move.call_count, 9)
        for direction in range(3):
            calls = self.move.call_args_list[3*direction:3*direction+3]
            self.assertEqual([c.kwargs["offset"] for c in calls], [.55, .60, .65])
        self.assertEqual(len([event for event in report["history"]
                              if event["event"] == "no_motion_route_unavailable"]), 3)

    def test_capture_timeout_falls_back_without_motion(self):
        self.effects.capture_lidar_view = Mock(side_effect=TourScanCaptureError("no fresh stopped scans"))
        _, report, hint = self.run_adapter()
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(report["history"][-1]["reason"], "fresh_scan_cohort_unavailable")
        self.move.assert_not_called()

    def test_missing_scan_mount_proof_keeps_alignment_unverified_without_probe_motion(self):
        self.effects.capture_lidar_view = Mock(side_effect=lambda request: replace(self.capture(request), mount_evidence=()))
        _, report, _ = self.run_adapter()
        self.assertFalse(report["head_alignment_verified"])
        self.assertTrue(report["history"][-1]["acquisition_unavailable"])
        self.assertFalse(report["history"][-1]["head_observability"]["accepted"])
        self.effects.capture_lidar_view.assert_called_once()
        self.move.assert_not_called()

    def test_incompatible_stand_model_never_starts_scan_capture_or_motion(self):
        self.source.measured_stand_model = replace(self.source.measured_stand_model, head_width_m=.09)
        self.effects.capture_lidar_view = Mock(side_effect=AssertionError("incompatible model must not start capture"))
        _, report, hint = self.run_adapter()
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertFalse(report["model_admission"]["accepted"])
        self.assertEqual(report["history"], [])
        self.fresh.assert_not_called()
        self.effects.capture_lidar_view.assert_not_called()
        self.move.assert_not_called()

    def test_capture_binding_mismatch_is_not_treated_as_route_retry(self):
        self.effects.capture_lidar_view = lambda request: replace(self.capture(request), candidate_uid="other")
        with self.assertRaisesRegex(ValueError, "capture candidate binding mismatch"):
            self.run_adapter()
        self.move.assert_not_called()

    def test_capture_execution_error_propagates_without_motion(self):
        self.effects.capture_lidar_view = Mock(side_effect=RuntimeError("capture evidence corrupt"))
        with self.assertRaisesRegex(RuntimeError, "capture evidence corrupt"):
            self.run_adapter()
        self.move.assert_not_called()


class CandidateLidarArrivalHandoffTest(unittest.TestCase):
    def run_handoff(self, root, *, verified):
        initial, registry = frame_fixture()
        # Distinguish the freshly measured arrival from the input and legacy
        # map-centroid admission so a mistaken branch is observable.
        fresh = replace(initial, observation_pose=initial.planning_frame.current_pose)
        legacy = replace(initial, observation_pose=replace(initial.planning_frame.current_pose, x_m=-.59))
        source = SimpleNamespace(camera_calibration=calibration(), max_candidate_inspection_views=8)
        effects = SimpleNamespace(capture_lidar_view=Mock(), admit_planning_frame=Mock(),
            load_route_uncertainty_readiness=Mock(), event_sink=Mock(), clock=lambda: 30.,
            run_centering_turn=None)
        admit = Mock(side_effect=AssertionError("verified current head must bypass old centroid yaw")
                     if verified else None, return_value=legacy)
        prepare_result = (fresh, {"head_alignment_verified": verified,
                                 "camera_centered_verified": False}, object())
        sentinel = object()
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.prepare_lidar_camera_arrival",
                   return_value=prepare_result) as prepare, patch(
                "scripts.aufgabe04.real_robot.candidate.inspection_adapters.execute_candidate_inspection",
                return_value=sentinel) as execute:
            result = execute_local_candidate_inspection(
                observation_frame=initial, source_config=source, effects=effects, source_registry=registry,
                candidate_root=root, candidate_run_id="run", candidate_index=0,
                admit_arrival=admit, admit_planning=Mock(side_effect=AssertionError("unexpected planning")),
                move_certified_opposite=Mock(side_effect=AssertionError("unexpected opposite motion")),
                execute_motion=Mock(side_effect=AssertionError("unexpected motion")),
                frame_type=_CandidateObservationFrame, request_type=SimpleNamespace,
                observation_request_type=SimpleNamespace)
        self.assertIs(result, sentinel)
        prepare.assert_called_once()
        execute.assert_called_once()
        return admit, execute.call_args.kwargs["initial_frame"], fresh, legacy

    def test_verified_head_arrival_bypasses_old_centroid_yaw_but_keeps_visual_centering_unverified(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            admit, received, fresh, _ = self.run_handoff(root, verified=True)
            admit.assert_not_called()
            self.assertIs(received, fresh)
            receipt = load_content_hashed_json(root / "candidate_arrival_admission.json",
                                              hash_field="candidate_arrival_admission_sha256")
            self.assertTrue(receipt["accepted"])
            self.assertTrue(receipt["head_alignment_verified"])
            self.assertFalse(receipt["camera_centered"])
            self.assertFalse(receipt["camera_centered_verified"])
            self.assertTrue(receipt["requires_live_target_association"])
            self.assertFalse(receipt["motion_authorized"])
            self.assertEqual(receipt["admission_kind"], "fresh_lidar_calibrated_camera_alignment")

    def test_unverified_lidar_arrival_still_requires_ordinary_arrival_admission(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            admit, received, _, legacy = self.run_handoff(root, verified=False)
            admit.assert_called_once()
            self.assertIs(received, legacy)
            self.assertFalse((root / "candidate_arrival_admission.json").exists())


if __name__ == "__main__":
    unittest.main()
