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
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid_with_bundle
from scripts.aufgabe04.perception.stand_axis.model_profile import load_measured_physical_stand_model
from scripts.aufgabe04.real_robot.candidate.approach import _CandidateObservationFrame
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import CandidateInspectionRouteUnavailableError
from scripts.aufgabe04.real_robot.candidate.observation_deferral import CandidateObservationUnavailableError
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition import (
    create_lidar_camera_recovery, create_bounded_lidar_recovery,
)
from scripts.aufgabe04.real_robot.candidate.lidar_acquisition_capture import (
    CandidateLidarView, CandidateLidarCaptureUnavailableError,
)
from scripts.aufgabe04.real_robot.readiness.tour_scan_capture import TourScanCaptureError
from tests.aufgabe04.test_lidar_alignment_arrival import calibration
from tests.aufgabe04.test_lidar_inspection_hint import hint_fixture, line_receipts
from tests.aufgabe04 import test_candidate_preapproach_planning as planning_fixtures
from tests.aufgabe04.test_detected_station_exploration import write_free_map


def frame_fixture():
    snapshot, registry, planning = hint_fixture()
    return _CandidateObservationFrame(candidate=snapshot.candidates[0], config=SimpleNamespace(snapshot=snapshot),
        planning_frame=planning, decision_binding=SimpleNamespace(
            projection_path=Path("candidate_frame_projection.json"), projection_sha256="c"*64)), registry


class BoundedLidarAcquisitionTest(unittest.TestCase):
    def controller(self, *, hint=None, current_fit=None, verified=False, centered=True,
                   unavailable=False, retryable=False, motion_error=None, budget_sec=120.):
        frame, _ = frame_fixture()
        state = dict(hint=hint, verified=verified, centered=centered, unavailable=unavailable,
                     retryable=retryable,
                     now=0., observation_cost=0., motion_cost=0., preflight_cost=0.)
        calls, revisions = [], []

        def observe(current, serial, checkpoint):
            calls.append(("observe", serial))
            state["now"] += state["observation_cost"]
            return current, state["hint"], current_fit, {
                "head_alignment_verified": state["verified"],
                "camera_centered_verified": state["centered"] and state["verified"],
                "acquisition_unavailable": state["unavailable"],
                "acquisition_retryable": state["retryable"],
            }

        def move(kind, current, target, serial, checkpoint):
            state["now"] += state["preflight_cost"]
            checkpoint("motion_dispatch")
            calls.append((kind, serial, target))
            state["now"] += state["motion_cost"]
            if motion_error is not None:
                raise motion_error
            return current

        recovery = create_bounded_lidar_recovery(observe=observe,
            move_probe=lambda *args: move("probe", *args),
            move_aligned=lambda *args: move("aligned", *args),
            persist=lambda report: revisions.append(copy.deepcopy(report)),
            monotonic=lambda: state["now"], budget_sec=budget_sec)
        return recovery, frame, state, calls, revisions

    def test_each_call_yields_to_camera_after_one_move_and_fresh_observation(self):
        recover, frame, _, calls, _ = self.controller()
        for serial in range(1, 4):
            before = len(calls)
            frame, report, _ = recover(frame)
            self.assertTrue(report["motion_completed"])
            self.assertEqual([c[0] for c in calls[before:]], ["observe", "probe", "observe"])
            self.assertEqual(report["probe_moves_attempted"], serial)
        _, report, _ = recover(frame)
        self.assertFalse(report["motion_completed"])
        self.assertEqual(report["reason"], "independent_geometry_support_unavailable")
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual([c[1] for c in calls if c[0] == "probe"], [1, 2, 3])

    def test_no_motion_rejections_consume_all_proposal_slots_without_reset(self):
        error = CandidateInspectionRouteUnavailableError("blocked", reason_code="static_clearance")
        recover, frame, _, calls, revisions = self.controller(motion_error=error)
        _, report, _ = recover(frame)
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 3)
        self.assertFalse(report["motion_completed"])
        self.assertEqual(len([e for e in report["history"] if e["event"] == "no_motion_route_unavailable"]), 3)
        proposals = [r for r in revisions if r["history"][-1]["event"] == "motion_proposal"]
        self.assertEqual([r["probe_moves_attempted"] for r in proposals], [1, 2, 3])
        before = len(calls)
        recover(frame)
        self.assertEqual(len(calls), before)

    def test_alignment_budget_persists_across_calls_and_known_no_motion_failure(self):
        for error in (None, CandidateInspectionRouteUnavailableError("blocked")):
            with self.subTest(error=error):
                recover, frame, _, calls, _ = self.controller(hint=object(), motion_error=error)
                for _ in range(3):
                    frame, report, _ = recover(frame)
                self.assertEqual([c[1] for c in calls if c[0] == "aligned"], [1, 2])
                self.assertEqual(report["alignment_moves_attempted"], 2)
                self.assertEqual(report["reason"], "alignment_correction_budget_exhausted")

    def test_alternating_fit_availability_never_resets_either_budget(self):
        recover, frame, state, calls, _ = self.controller()
        for hint in (None, object(), None, object(), None):
            state["hint"] = hint
            frame, report, _ = recover(frame)
            self.assertTrue(report["motion_completed"])
        _, report, _ = recover(frame)
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 3)
        self.assertEqual(len([c for c in calls if c[0] == "aligned"]), 2)
        self.assertEqual(report["probe_moves_attempted"], 3)
        self.assertEqual(report["alignment_moves_attempted"], 2)

    def test_verified_arrival_and_unavailable_capture_never_send_motion(self):
        for values in (dict(hint=object(), verified=True), dict(unavailable=True)):
            recover, frame, _, calls, _ = self.controller(**values)
            _, report, _ = recover(frame)
            self.assertEqual(calls, [("observe", 0)])
            self.assertFalse(report["motion_completed"])
            self.assertFalse(report["motion_authorized"])

    def test_single_view_fit_is_probe_hint_never_alignment_authority(self):
        fit = SimpleNamespace(normals=lambda *_: (0., math.pi))
        recover, frame, _, calls, _ = self.controller(current_fit=fit)
        _, report, _ = recover(frame)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 1)
        self.assertFalse(any(c[0] == "aligned" for c in calls))

    def test_retryable_acquisition_yields_without_motion_or_resetting_time_budget(self):
        recover, frame, state, calls, _ = self.controller(
            hint=object(), retryable=True, budget_sec=3.)
        state["observation_cost"] = 1.
        for _ in range(3):
            _, report, hint = recover(frame)
            self.assertFalse(report["motion_completed"])
            self.assertFalse(report["recovery_complete"])
            self.assertIsNone(hint)
        _, report, _ = recover(frame)
        self.assertEqual(report["reason"], "lidar_recovery_time_budget_exhausted")
        self.assertTrue(report["recovery_complete"])
        self.assertEqual(calls, [("observe", 0), ("observe", 1), ("observe", 2)])

    def test_invalid_post_motion_observation_remains_terminal_on_later_call(self):
        frame, _ = frame_fixture()
        observe = Mock(side_effect=[(frame, None, None, {}), (frame, None, None, {
            "acquisition_unavailable": True, "reason": "invalid_mount"})])
        move = Mock(return_value=frame)
        recover = create_bounded_lidar_recovery(observe=observe, move_probe=move,
            move_aligned=move, persist=lambda _: None)
        _, first, _ = recover(frame)
        _, second, _ = recover(frame)
        self.assertTrue(first["motion_completed"])
        self.assertTrue(first["recovery_complete"])
        self.assertTrue(second["recovery_complete"])
        self.assertEqual(second["reason"], "invalid_mount")
        self.assertEqual(observe.call_count, 2)
        move.assert_called_once()

    def test_active_time_accumulates_but_camera_time_between_calls_is_excluded(self):
        recover, frame, state, calls, _ = self.controller(budget_sec=20.)
        state.update(observation_cost=1., motion_cost=5.)
        frame, first, _ = recover(frame)
        state["now"] += 1000.  # Camera work does not consume recovery's budget.
        frame, second, _ = recover(frame)
        self.assertEqual(first["active_elapsed_sec"], 7.)
        self.assertEqual(second["active_elapsed_sec"], 14.)
        self.assertEqual(len([c for c in calls if c[0] == "probe"]), 2)
        before_expiry = len(calls)
        frame, third, _ = recover(frame)
        self.assertTrue(third["motion_completed"])
        self.assertFalse(third["head_alignment_verified"])
        self.assertEqual([c[0] for c in calls[before_expiry:]], ["observe", "probe"])
        self.assertEqual(third["reason"], "lidar_recovery_time_budget_exhausted")
        before = len(calls)
        recover(frame)
        self.assertEqual(len(calls), before)

    def test_deadline_rechecked_after_planning_before_motion_dispatch(self):
        recover, frame, state, calls, _ = self.controller(budget_sec=2.)
        state["preflight_cost"] = 3.
        _, report, _ = recover(frame)
        self.assertFalse(report["motion_completed"])
        self.assertEqual(report["reason"], "lidar_recovery_time_budget_exhausted")
        self.assertFalse(any(c[0] == "probe" for c in calls))
        self.assertEqual(report["probe_moves_attempted"], 1)

    def test_wrapped_deadline_error_does_not_consume_another_proposal(self):
        recover, frame, state, calls, _ = self.controller(
            budget_sec=2., motion_error=CandidateInspectionRouteUnavailableError(
                "wrapped pre-dispatch expiry", reason_code="standoff_proposals_exhausted"))
        state["motion_cost"] = 3.
        _, report, _ = recover(frame)
        self.assertEqual(report["reason"], "lidar_recovery_time_budget_exhausted")
        self.assertEqual(report["probe_moves_attempted"], 1)
        self.assertFalse(report["motion_completed"])

    def test_shared_route_budget_exhaustion_terminates_without_more_proposals(self):
        recover, frame, _, calls, _ = self.controller(
            motion_error=CandidateInspectionRouteUnavailableError(
                "global ledger exhausted", reason_code="route_proposal_budget_exhausted"))
        _, report, _ = recover(frame)
        self.assertEqual(report["reason"], "route_proposal_budget_exhausted")
        self.assertEqual(report["probe_moves_attempted"], 1)
        self.assertFalse(report["motion_completed"])
        self.assertTrue(report["recovery_complete"])

    def test_execution_errors_and_interruptions_propagate_and_record_boundary(self):
        for hint in (None, object()):
            for error in (RuntimeError("motion outcome unknown"), KeyboardInterrupt()):
                with self.subTest(hint=hint, error=type(error)):
                    recover, frame, _, calls, revisions = self.controller(hint=hint, motion_error=error)
                    with self.assertRaises(type(error)):
                        recover(frame)
                    self.assertEqual(sum(c[0] in ("probe", "aligned") for c in calls), 1)
                    self.assertIn(revisions[-1]["history"][-1]["event"], ("recovery_failed", "recovery_interrupted"))


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
            camera_arrival_range_slack_m=.20, camera_arrival_max_bearing_error_rad=math.radians(3.))
        self.move = Mock(side_effect=AssertionError("verified/unavailable arrival must not move"))
        self.fresh = Mock(return_value=self.frame)
        self.uncertainty = SimpleNamespace(covariance=PlanarCovariance(1e-6, 0., 1e-6),
            admission_config=SimpleNamespace(localization_sigma_multiplier=1., heading_sigma_rad=math.radians(.1)))
        self.now = 30.
        self.capture_distance = .6
        self.capture_requests = []
        self.effects = SimpleNamespace(clock=lambda: self.now, capture_lidar_view=self.capture)

    def capture(self, request):
        self.capture_requests.append(request)
        start = request.observation_not_before_sec
        receipts = tuple(replace(r, receipt_id=f"arrival_{len(self.capture_requests)}_{i}", viewpoint_id=request.viewpoint_id,
                                 scan_stamp_sec=start+.05+i*.08, pose_stamp_sec=start+.05+i*.08,
                                 observer_clock_sec=start+.06+i*.08)
                         for i,r in enumerate(line_receipts(increment=.01, distance=self.capture_distance)[:3]))
        self.now = start+.3
        return CandidateLidarView(request.candidate_uid, request.candidate_snapshot_sha256,
            request.viewpoint_id, receipts, receipts[-1].frame_provenance.canonical_scan_pose_odom,
            receipts[-1].scan_stamp_sec, start+.3, request.output_dir / "candidate_lidar_view.json", "a"*64,
            request.output_dir / "cohort.json", "b"*64,
            tuple({"stamp_sec": r.scan_stamp_sec, "ground_frame": "base_footprint",
                   "scan_height_above_ground_m": .182, "scan_vertical_direction_x": 0.,
                   "scan_vertical_direction_y": 0., "scan_vertical_direction_z": 1.,
                   "exact_transform_stamp_sec": r.scan_stamp_sec} for r in receipts))

    def make_adapter(self, **kwargs):
        return create_lidar_camera_recovery(source_config=self.source,
            source_registry=self.registry, effects=self.effects, candidate_root=self.root,
            fresh_frame=self.fresh, plan_and_move=self.move,
            load_uncertainty=lambda _: self.uncertainty, **kwargs)

    def run_adapter(self, *, survey=None):
        survey = (line_receipts(increment=.01) + line_receipts(2, increment=.01)) if survey is None else survey
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.load_camera_lidar_receipts",
                   return_value=(survey, {})):
            return self.make_adapter()(self.frame)

    def test_construction_never_captures_plans_or_writes_before_camera(self):
        recover = self.make_adapter()
        self.assertTrue(callable(recover))
        self.fresh.assert_not_called()
        self.move.assert_not_called()
        self.assertEqual(self.capture_requests, [])
        self.assertFalse((self.root / "lidar_head_acquisition").exists())

    def test_real_fits_verify_and_persist_hashed_report(self):
        returned, report, hint = self.run_adapter()
        self.assertIsNotNone(hint)
        self.assertTrue(report["head_alignment_verified"], report)
        self.assertTrue(report["camera_centered_verified"])
        self.move.assert_not_called()
        saved = load_content_hashed_json(Path(report["arrival_review_path"]),
                                        hash_field="lidar_alignment_arrival_sha256")
        current = next(e for e in reversed(saved["history"]) if e["event"] == "stopped_observation")
        self.assertEqual(current["reason"], "fresh_lidar_camera_alignment_verified")
        self.assertIn("camera_calibration_sha256", current)
        self.assertEqual(len(current["source_receipt_sha256s"]), 3)
        self.assertEqual(returned.observation_pose, returned.planning_frame.current_pose)
        self.assertIsNotNone(returned.camera_target_geometry)
        # The fixture face is x=0; the fitted solid center is half its depth behind it.
        self.assertAlmostEqual(returned.camera_target_geometry.x_m,
                               self.source.measured_stand_model.head_depth_m / 2., places=6)
        self.assertAlmostEqual(returned.camera_target_geometry.y_m, 0., places=6)
        self.assertEqual(returned.candidate, self.frame.candidate)

    def test_single_supported_view_rejects_redundant_probe_then_bounds_other_routes(self):
        self.capture_distance = .58
        self.move.side_effect = CandidateInspectionRouteUnavailableError("unsafe static route")
        returned, report, hint = self.run_adapter(survey=())
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertIsNone(returned.camera_target_geometry)
        self.assertEqual(report["probe_moves_attempted"], 3)
        self.assertEqual(self.move.call_count, 6)  # .58→.55 m cannot add 5 cm of range support.
        self.assertEqual([c.kwargs["offset"] for c in self.move.call_args_list], [.55, .60, .65] * 2)
        rejected = [e for e in report["history"] if e["event"] == "no_motion_route_unavailable"]
        self.assertEqual(rejected[0]["reason"], "redundant_lidar_support_probe")
        self.assertTrue(all("before_motion" in c.kwargs for c in self.move.call_args_list))

    def test_real_fit_recovery_retains_receipts_but_yields_after_one_move_per_call(self):
        self.capture_distance = .58
        def move(frame, *args, **kwargs):
            kwargs["before_motion"]()
            return frame
        self.move.side_effect = move
        recover = self.make_adapter()
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.load_camera_lidar_receipts",
                   return_value=((), {})) as load:
            frame, first, hint = recover(self.frame)
            self.assertTrue(first["motion_completed"])
            self.assertEqual(self.move.call_count, 1)
            self.assertEqual(len(self.capture_requests), 2)  # Before and after displacement.
            self.assertIsNone(hint)  # Mock motion cannot manufacture an independent view.
            _, second, _ = recover(frame)
        self.assertTrue(second["motion_completed"])
        self.assertEqual(self.move.call_count, 2)
        self.assertEqual(second["probe_moves_attempted"], 3)  # Initial redundant proposal consumed one slot.
        self.assertEqual(len(self.capture_requests), 4)
        load.assert_called_once()
        stopped = [e for e in second["history"] if e["event"] == "stopped_observation"]
        self.assertEqual(len(stopped), 4)
        self.assertEqual(len(stopped[-1]["fit_diagnostics"]["scan_count_by_view"]), 4)

    def test_same_bearing_probe_can_still_move_closer_when_range_gain_is_useful(self):
        self.capture_distance = .65
        self.move.side_effect = CandidateInspectionRouteUnavailableError("unsafe static route")
        _, report, _ = self.run_adapter(survey=())
        self.assertEqual(self.move.call_count, 9)
        self.assertFalse(any(e.get("reason") == "redundant_lidar_support_probe" for e in report["history"]))

    def test_configured_half_meter_probe_is_not_rejected_using_old_point_five_five_threshold(self):
        self.source.approach_offset_m = .50
        self.capture_distance = .58
        self.move.side_effect = CandidateInspectionRouteUnavailableError("unsafe static route")
        _, report, _ = self.run_adapter(survey=())
        self.assertEqual([c.kwargs["offset"] for c in self.move.call_args_list], [.50, .55, .60] * 3)
        self.assertFalse(any(e.get("reason") == "redundant_lidar_support_probe" for e in report["history"]))

    def test_partial_two_of_three_fit_recovers_with_another_cohort_same_view(self):
        def capture(request):
            result = self.capture(request)
            if len(self.capture_requests) == 1:
                bad = replace(result.receipts[-1], ranges_m=(None,) * len(result.receipts[-1].ranges_m))
                result = replace(result, receipts=result.receipts[:-1]+(bad,))
            return result
        self.effects.capture_lidar_view = capture
        _, report, hint = self.run_adapter(survey=line_receipts(2, increment=.01))
        self.assertTrue(report["head_alignment_verified"])
        self.assertEqual(len(self.capture_requests), 2)
        self.assertEqual(len({r.viewpoint_id for r in self.capture_requests}), 1)
        self.assertEqual(len({r.output_dir for r in self.capture_requests}), 2)
        view_id = self.capture_requests[0].viewpoint_id
        self.assertEqual(hint.evidence["scan_count_by_view"][view_id], 5)
        event = next(e for e in reversed(report["history"]) if e["event"] == "stopped_observation")
        self.assertEqual(event["examined_scan_count"], 6)
        self.assertEqual(len(event["source_receipt_sha256s"]), 3)  # Fresh verifier never consumes the older cohort.
        self.move.assert_not_called()

    def test_bad_first_cohort_counts_against_all_subsequent_support(self):
        def capture(request):
            result = self.capture(request)
            if len(self.capture_requests) == 1:
                result = replace(result, receipts=tuple(replace(r, ranges_m=(None,) * len(r.ranges_m)) for r in result.receipts))
            return result
        self.effects.capture_lidar_view = capture
        self.move.side_effect = CandidateInspectionRouteUnavailableError("unsafe static route")
        _, report, hint = self.run_adapter(survey=line_receipts(2, increment=.01))
        self.assertIsNone(hint)  # 6/9 is below 75%; no discarding the initial failures.
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(len(self.capture_requests), 3)
        event = next(e for e in reversed(report["history"]) if e["event"] == "stopped_observation")
        self.assertEqual(event["examined_scan_count"], 9)
        local = self.capture_requests[0].viewpoint_id
        self.assertEqual(event["fit_diagnostics"]["scan_count_by_view"][local], 9)
        self.assertEqual(event["fit_diagnostics"]["surface_fit_count_by_view"][local], 6)

    def test_additional_cohorts_require_the_same_stationary_view(self):
        def capture(request):
            result = self.capture(request)
            if len(self.capture_requests) == 1:
                bad = replace(result.receipts[-1], ranges_m=(None,) * len(result.receipts[-1].ranges_m))
                return replace(result, receipts=result.receipts[:-1]+(bad,))
            if len(self.capture_requests) == 2:
                shifted = tuple(replace(r, scan_pose_map=replace(r.scan_pose_map, y_m=.04),
                    frame_provenance=replace(r.frame_provenance,
                    canonical_scan_pose_odom=replace(r.frame_provenance.canonical_scan_pose_odom, y_m=.04)))
                    for r in result.receipts)
                return replace(result, receipts=shifted)
            return result
        self.effects.capture_lidar_view = capture
        self.move.side_effect = CandidateInspectionRouteUnavailableError("unsafe static route")
        _, report, hint = self.run_adapter(survey=line_receipts(2, increment=.01))
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(len(self.capture_requests), 3)
        event = next(e for e in reversed(report["history"]) if e["event"] == "stopped_observation")
        self.assertFalse(event["stopped_view_fit_usable"])

    def test_second_cohort_mount_must_be_admitted_before_use(self):
        def capture(request):
            result = self.capture(request)
            if len(self.capture_requests) == 1:
                bad = replace(result.receipts[-1], ranges_m=(None,) * len(result.receipts[-1].ranges_m))
                return replace(result, receipts=result.receipts[:-1]+(bad,))
            return replace(result, mount_evidence=())
        self.effects.capture_lidar_view = capture
        _, report, hint = self.run_adapter(survey=line_receipts(2, increment=.01))
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(len(self.capture_requests), 2)
        self.assertTrue(report["history"][-1]["acquisition_unavailable"])
        self.move.assert_not_called()

    def test_capture_timeout_falls_back_without_motion(self):
        self.effects.capture_lidar_view = Mock(side_effect=TourScanCaptureError("no fresh stopped scans"))
        _, report, hint = self.run_adapter()
        self.assertIsNone(hint)
        self.assertFalse(report["head_alignment_verified"])
        self.assertEqual(report["history"][-1]["reason"], "fresh_scan_cohort_unavailable")
        self.assertFalse(report["recovery_complete"])
        self.assertTrue(report["history"][-1]["acquisition_retryable"])
        self.assertEqual(self.effects.capture_lidar_view.call_count, 3)
        self.assertEqual(len(report["history"][-1]["capture_failures"]), 3)
        self.move.assert_not_called()

    def test_timeout_and_stale_capture_retry_with_fresh_floor_and_same_viewpoint(self):
        for error in (TourScanCaptureError("exact-time TF unavailable",
                      diagnostics={"rejection_attempt_counts": {"exact_time_transform_unavailable": 4}}),
                      CandidateLidarCaptureUnavailableError("stale cohort")):
            with self.subTest(error=type(error).__name__):
                self.root = self.root / type(error).__name__
                self.now = 30.
                self.capture_requests = []
                def capture(request):
                    if self.effects.capture_lidar_view.call_count == 1:
                        self.now += 3.
                        raise error
                    return self.capture(request)
                self.effects.capture_lidar_view = Mock(side_effect=capture)
                _, report, hint = self.run_adapter()
                self.assertTrue(report["head_alignment_verified"])
                self.assertIsNotNone(hint)
                first, second = [c.args[0] for c in self.effects.capture_lidar_view.call_args_list]
                self.assertEqual(first.viewpoint_id, second.viewpoint_id)
                self.assertNotEqual(first.output_dir, second.output_dir)
                self.assertEqual(second.observation_not_before_sec-first.observation_not_before_sec, 3.)
                event = report["history"][-1]
                self.assertEqual(len(event["capture_failures"]), 1)
                self.assertEqual(event["capture_failures"][0]["diagnostics"], getattr(error, "diagnostics", {}))
                self.move.assert_not_called()

    def test_exhausted_passive_cohorts_do_not_disable_later_camera_view(self):
        attempts = 0
        def capture(request):
            nonlocal attempts
            attempts += 1
            if attempts <= 3:
                self.now += 3.
                raise TourScanCaptureError("exact-time TF unavailable")
            return self.capture(request)
        self.effects.capture_lidar_view = capture
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.load_camera_lidar_receipts",
                   return_value=(line_receipts(increment=.01)+line_receipts(2, increment=.01), {})):
            recover = self.make_adapter()
            frame, first, _ = recover(self.frame)
            _, second, hint = recover(frame)
        self.assertFalse(first["recovery_complete"])
        self.assertTrue(second["head_alignment_verified"])
        self.assertIsNotNone(hint)
        self.assertEqual(attempts, 4)
        self.assertEqual(second["probe_moves_attempted"], 0)
        self.move.assert_not_called()

    def test_failure_after_valid_cohort_does_not_probe_using_older_support(self):
        attempts = 0
        def capture(request):
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise TourScanCaptureError("no fresh cohort")
            result = self.capture(request)
            bad = replace(result.receipts[-1], ranges_m=(None,) * len(result.receipts[-1].ranges_m))
            return replace(result, receipts=result.receipts[:-1]+(bad,))
        self.effects.capture_lidar_view = capture
        _, report, hint = self.run_adapter(survey=line_receipts(2, increment=.01))
        self.assertIsNone(hint)
        self.assertEqual(report["probe_moves_attempted"], 0)
        self.assertTrue(report["history"][-1]["acquisition_retryable"])
        self.assertEqual(report["history"][-1]["examined_scan_count"], 3)
        self.move.assert_not_called()

    def test_nonretryable_capture_error_terminates_and_is_not_retried(self):
        self.effects.capture_lidar_view = Mock(side_effect=TourScanCaptureError(
            "ROS2 unavailable", retryable=False, reason_code="scan_capture_dependencies_unavailable"))
        recover = self.make_adapter()
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.load_camera_lidar_receipts",
                   return_value=((), {})):
            _, first, _ = recover(self.frame)
            _, second, _ = recover(self.frame)
        self.assertTrue(first["recovery_complete"])
        self.assertTrue(second["recovery_complete"])
        self.assertEqual(second["reason"], "scan_capture_dependencies_unavailable")
        self.effects.capture_lidar_view.assert_called_once()
        self.move.assert_not_called()

    def test_retryable_capture_attempts_stop_when_cumulative_active_budget_expires(self):
        attempts = 0
        def capture(request):
            nonlocal attempts
            attempts += 1
            self.now += 3.
            raise TourScanCaptureError("no fresh cohort")
        self.effects.capture_lidar_view = capture
        recover = self.make_adapter(monotonic=lambda: self.now, budget_sec=10.)
        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.load_camera_lidar_receipts",
                   return_value=((), {})):
            _, first, _ = recover(self.frame)
            self.now += 60.  # Camera time remains outside the LiDAR budget.
            _, second, _ = recover(self.frame)
            _, third, _ = recover(self.frame)
        self.assertFalse(first["recovery_complete"])
        self.assertEqual(first["active_elapsed_sec"], 9.)
        self.assertEqual(second["active_elapsed_sec"], 12.)
        self.assertEqual(second["reason"], "lidar_recovery_time_budget_exhausted")
        self.assertTrue(third["recovery_complete"])
        self.assertEqual(attempts, 4)
        self.move.assert_not_called()

    def test_missing_mount_proof_never_sends_probe_motion(self):
        self.effects.capture_lidar_view = Mock(side_effect=lambda request: replace(self.capture(request), mount_evidence=()))
        _, report, _ = self.run_adapter()
        self.assertFalse(report["head_alignment_verified"])
        self.assertTrue(report["history"][-1]["acquisition_unavailable"])
        self.assertFalse(report["history"][-1]["head_observability"]["accepted"])
        self.effects.capture_lidar_view.assert_called_once()
        self.move.assert_not_called()

    def test_incompatible_stand_model_never_starts_capture_or_motion(self):
        self.source.measured_stand_model = replace(self.source.measured_stand_model, head_width_m=.09)
        self.effects.capture_lidar_view = Mock(side_effect=AssertionError("incompatible model must not capture"))
        _, report, hint = self.run_adapter()
        self.assertIsNone(hint)
        self.assertFalse(report["model_admission"]["accepted"])
        self.fresh.assert_not_called()
        self.effects.capture_lidar_view.assert_not_called()
        self.move.assert_not_called()

    def test_corrupt_capture_identity_is_fatal_not_route_retry(self):
        self.effects.capture_lidar_view = lambda request: replace(self.capture(request), candidate_uid="other")
        with self.assertRaisesRegex(ValueError, "capture candidate binding mismatch"):
            self.run_adapter()
        self.move.assert_not_called()

    def test_unknown_capture_failure_propagates_without_motion(self):
        self.effects.capture_lidar_view = Mock(side_effect=RuntimeError("capture evidence corrupt"))
        with self.assertRaisesRegex(RuntimeError, "capture evidence corrupt"):
            self.run_adapter()
        self.move.assert_not_called()

    def test_direct_sampling_turn_rechecks_target_before_any_motion_dispatch(self):
        map_yaml = write_free_map(self.root)
        _, bundle = load_occupancy_grid_with_bundle(map_yaml, semantic_map_id="arena", planning_frame="map")
        candidate = replace(self.frame.candidate,
                            geometry=replace(self.frame.candidate.geometry, x_m=3.))
        snapshot = replace(self.frame.config.snapshot, map_bundle_sha256=bundle.bundle_sha256,
                           candidates=(candidate,))
        config = SimpleNamespace(snapshot=snapshot, map_yaml=map_yaml, semantic_map_id="arena",
            plan=planning_fixtures.CandidatePreapproachPlanningTest._plan(bundle.bundle_sha256))
        frame = replace(self.frame, config=config, candidate=candidate)
        self.effects.run_lidar_sampling_turn = Mock(side_effect=AssertionError("invalid target must not turn"))
        checkpoint = Mock()

        # Call the adapter's production sampling callback directly: even if an
        # upstream observation/controller gate were bypassed, dispatch is guarded.
        def bind_sampling(**callbacks):
            return lambda current: callbacks["move_sampling"](current, {}, 1, checkpoint)

        with patch("scripts.aufgabe04.real_robot.candidate.lidar_acquisition.create_bounded_lidar_recovery",
                   side_effect=bind_sampling):
            recovery = self.make_adapter()
            with self.assertRaises(CandidateObservationUnavailableError) as raised:
                recovery(frame)
        self.assertEqual(raised.exception.reason, "candidate_target_ineligible")
        self.assertEqual(raised.exception.status_evidence["reasons"], ["target_static_map_incompatible"])
        self.assertFalse(raised.exception.process_evidence["observer_started"])
        self.effects.run_lidar_sampling_turn.assert_not_called()
        self.move.assert_not_called()
        self.fresh.assert_not_called()
        self.assertEqual(self.capture_requests, [])
        checkpoint.assert_called_once_with("sampling_turn_preflight", sampling_index=1)
        evidence = load_content_hashed_json(
            self.root / "lidar_head_acquisition/sampling_01_target_admission.json",
            hash_field="candidate_target_admission_sha256")
        self.assertFalse(evidence["accepted"])
        self.assertFalse(evidence["motion_authorized"])
        self.assertFalse(evidence["keepouts_changed"])


if __name__ == "__main__":
    unittest.main()
