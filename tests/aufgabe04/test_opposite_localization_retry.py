"""Bounded no-motion recovery for the 20260928 opposite-face failure."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.real_robot.candidate import approach
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    CandidateInspectionRouteUnavailableError, bounded_inspection_standoffs,
)
from scripts.aufgabe04.real_robot.candidate.opposite_localization_retry import (
    OPPOSITE_UNCERTAINTY_EXHAUSTED, with_opposite_localization_retry,
)
from scripts.aufgabe04.real_robot.candidate.recovery_failure import CandidateStartupRecoveryError
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.approach.candidate_preapproach_compute import validate_approach_outside_transit_keepout
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures
from tests.aufgabe04.test_detected_station_exploration import write_free_map


class OppositeLocalizationRetryTest(unittest.TestCase):
    @staticmethod
    def exhausted(**overrides):
        return CandidateInspectionRouteUnavailableError(
            "route margin negative", reason_code=OPPOSITE_UNCERTAINTY_EXHAUSTED,
            evidence={"motion_published": False, "motion_permit_issued": False,
                      "no_motion_uncertainty_rejections_verified": True, **overrides},
        )

    def test_refresh_can_converge_once_and_does_not_repeat_success(self):
        attempt = Mock(side_effect=[self.exhausted(), "arrived"])
        events = []
        self.assertEqual(with_opposite_localization_retry(
            attempt=attempt, enabled=True, event_sink=events.append), "arrived")
        self.assertEqual([c.args for c in attempt.call_args_list], [(0,), (1,)])
        self.assertEqual(events[0]["event"], "opposite_localization_refresh_requested")
        self.assertFalse(events[0]["motion_authorized"])

    def test_unchanged_uncertainty_exhausts_after_one_refresh_with_both_reasons(self):
        attempt = Mock(side_effect=self.exhausted())
        with self.assertRaises(CandidateInspectionRouteUnavailableError) as caught:
            with_opposite_localization_retry(attempt=attempt, enabled=True, event_sink=lambda _: None)
        self.assertEqual(attempt.call_count, 2)
        self.assertEqual(len(caught.exception.evidence["planning_epoch_failures"]), 2)

    def test_no_refresh_without_stationary_effect_or_complete_no_motion_evidence(self):
        for enabled, overrides in [(False, {}), (True, {"motion_published": True}),
                                   (True, {"motion_permit_issued": True}),
                                   (True, {"no_motion_uncertainty_rejections_verified": False})]:
            with self.subTest(enabled=enabled, overrides=overrides):
                attempt = Mock(side_effect=self.exhausted(**overrides))
                with self.assertRaises(CandidateInspectionRouteUnavailableError):
                    with_opposite_localization_retry(attempt=attempt, enabled=enabled, event_sink=lambda _: None)
                attempt.assert_called_once_with(0)

    def test_malformed_evidence_and_runtime_failures_remain_terminal(self):
        for error in [ValueError("receipt hash mismatch"), RuntimeError("motion failed")]:
            attempt = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                with_opposite_localization_retry(attempt=attempt, enabled=True, event_sink=lambda _: None)
            attempt.assert_called_once_with(0)

    def test_shared_floor_includes_retained_center_uncertainty(self):
        values = bounded_inspection_standoffs(.5, minimum_active_standoff_m=.33,
            candidate_transit_radius_m=.34, map_resolution_m=.05,
            target_center_uncertainty_m=.02512049541285534)
        self.assertEqual(values[:2], (.5, .45))
        self.assertGreater(values[-1], .4)
        self.assertLess(values[-1], .41)
        for value in values:
            validate_approach_outside_transit_keepout(approach_offset_m=value,
                candidate_transit_radius_m=.34, map_resolution_m=.05)
        self.assertNotIn(.35, values)
        self.assertEqual(bounded_inspection_standoffs(.4, minimum_active_standoff_m=.33,
            candidate_transit_radius_m=.34, map_resolution_m=.05,
            target_center_uncertainty_m=.1), ())
        for bad in [-1, float('nan'), True]:
            with self.assertRaises(ValueError):
                bounded_inspection_standoffs(.5, minimum_active_standoff_m=.33,
                    candidate_transit_radius_m=.34, map_resolution_m=.05,
                    target_center_uncertainty_m=bad)

    def test_recorded_failure_chain_uses_fresh_epoch_ids_and_returns_local_failure(self):
        self.exercise_coordinator()

    def test_recorded_chain_with_center_uncertainty_filters_point_four(self):
        self.exercise_coordinator(center_uncertainty=.02512049541285534)

    def test_child_with_motion_or_permit_cannot_refresh(self):
        for kwargs in [dict(motion_published=True), dict(report_mission_leg_permit=True)]:
            with self.subTest(kwargs=kwargs):
                self.exercise_coordinator(rejection_kwargs=kwargs)

    def exercise_coordinator(self, *, center_uncertainty=0.0, rejection_kwargs=None):
        factory = fixtures.AutonomousCandidateApproachTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = factory._candidate('candidate_1', .2, 0.)
            config = replace(factory._config(root, (candidate,)), approach_offset_m=.5,
                candidate_transit_radius_m=.34,
                physical_clearance={"minimum_active_standoff_m":.33,
                    "minimum_collision_standoff_m":.28,
                    "minimum_candidate_transit_radius_m":.34,"minimum_static_inflation_m":.25})
            write_free_map(root, resolution=.05)
            frame = approach._CandidateObservationFrame(config=config, candidate=candidate,
                planning_frame=None, decision_binding=None)
            source_receipt = root/'original_backside.json'
            observation = approach.CandidateObservation(None, None, source_receipt)
            plans, children, epochs, events = [], [], [], []
            def admit(**kw):
                epochs.append(kw['candidate_root'])
                return config, candidate, Pose2D(0,0,0), None, None
            def plan(request):
                plans.append(request)
                validate_approach_outside_transit_keepout(approach_offset_m=request.approach_offset_m,
                    candidate_transit_radius_m=.34, map_resolution_m=.05)
                if request.approach_offset_m < .45:
                    raise approach.CandidatePreapproachUnreachableError(candidate.candidate_uid,
                        '(25,24):goal_cell_not_traversable')
                return {'route_csv': str(request.output_dir/'route.csv')}
            def run(request):
                children.append(request)
                return factory._route_uncertainty_rejection(request, **(rejection_kwargs or {}))
            effects = approach.CandidateApproachEffects(read_current_pose=lambda:Pose2D(0,0,0),
                run_motion_leg=run, capture_observation=Mock(), plan_preapproach=plan,
                admit_planning_frame=Mock(), event_sink=lambda _, e:events.append(e))
            center = None if not center_uncertainty else dict(x_m=.2,y_m=0,uncertainty_m=center_uncertainty)
            with patch.object(approach, '_admit_opposite_face_planning_geometry', side_effect=admit), \
                 patch.object(approach, 'opposite_face_normal', return_value=1.2), \
                 patch.object(approach, 'load_backside_axis_planning_observation',
                              return_value=SimpleNamespace(validated_target_center=center)):
                error = CandidateStartupRecoveryError if rejection_kwargs else CandidateInspectionRouteUnavailableError
                with self.assertRaises(error) as caught:
                    approach._move_certified_opposite_face(observation_frame=frame, observation=observation,
                        source_config=config, effects=effects, source_registry=None,
                        candidate_root=root/'opposite',candidate_run_id='mission_inspect',candidate_index=1)
            if rejection_kwargs:
                self.assertEqual(len(epochs), 1)
                self.assertEqual(len(children), 1)
                return
            self.assertEqual(epochs, [root/'opposite', root/'opposite/localization_001'])
            self.assertEqual(len(children), 4)  # only .50/.45 each epoch reached dry admission
            self.assertEqual(len({r.run_id for r in children}), 4)
            self.assertEqual(len({r.permit_json_path for r in children}), 4)
            self.assertTrue(all(r.axis_observation_path == source_receipt for r in plans))
            self.assertTrue(all(r.approach_normal_rad == 1.2 for r in plans))
            self.assertTrue(all(r.approach_offset_m > .375355 for r in plans))
            self.assertEqual(len(caught.exception.evidence['planning_epoch_failures']), 2)
            self.assertEqual(caught.exception.reason_code, OPPOSITE_UNCERTAINTY_EXHAUSTED)
            self.assertEqual(sum(e['event']=='opposite_localization_refresh_requested' for e in events), 1)
            effects.capture_observation.assert_not_called()


class RecordedOppositeGeometryTest(unittest.TestCase):
    """Use the saved map and recorded geometry; no observer/ROS replay claims."""
    def test_blocked_inner_goal_and_outer_uncertainty_deficit_remain_enforced(self):
        import json
        import math
        from scripts.aufgabe04.navigation.approach import candidate_preapproach_compute as planner
        from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
        from scripts.aufgabe04.navigation.foundation.models import GridCell
        from scripts.aufgabe04.stations.candidate_snapshot import CandidateGeometry, new_candidate_snapshot
        from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import (
            RouteUncertaintyAdmissionConfig, evaluate_route_uncertainty_admission,
        )
        from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
        repo = Path(__file__).resolve().parents[2]
        d = json.loads((Path(__file__).parent/'fixtures/opposite_route_20260928.json').read_text())
        factory = fixtures.AutonomousCandidateApproachTest()
        with tempfile.TemporaryDirectory() as tmp:
            candidates = tuple(replace(factory._candidate(g['candidate_uid'],g['x_m'],g['y_m']),
                geometry=CandidateGeometry(**{k:v for k,v in g.items() if k!='candidate_uid'}))
                for g in d['candidate_geometry'])
            config = factory._config(Path(tmp), candidates)
            snapshot = new_candidate_snapshot(snapshot_id='recorded_geometry', created_unix_sec=1.,
                planning_frame='map',map_bundle_sha256=d['map_bundle_sha256'],candidates=candidates)
            plan = replace(config.plan,map_bundle_sha256=d['map_bundle_sha256'],arena_bounds=ArenaBounds(**d['arena_bounds']))
            kwargs = dict(map_yaml=repo/'maps/aufgabe03/arena_1p898x3p9_auto.yaml',
                semantic_map_id='arena_1p898x3p9_auto',plan=plan,snapshot=snapshot,
                inflation_radius_m=.25,candidate_transit_radius_m=.34,
                physical_clearance=d['physical_clearance'],validated_target_center=d['validated_target_center'])
            context = planner.load_candidate_planning_context(**kwargs)
            self.assertEqual(context.map_bundle.bundle_sha256,d['map_bundle_sha256'])
            self.assertEqual(context.costmaps.planning_costmap.cell_sources[GridCell(25,24)],'station_keepout')
            radius = .34 + d['validated_target_center']['uncertainty_m']
            self.assertEqual(math.ceil(radius/.05),8)
            with patch.object(planner,'load_candidate_planning_context',return_value=context):
                def compute(offset):
                    return planner.compute_candidate_preapproach_plan(**kwargs,
                        candidate_uid='survey_candidate_0001',start=Pose2D(**d['start']),
                        approach_offset_m=offset,approach_normal_rad=d['normal_rad'])
                with self.assertRaisesRegex(approach.CandidatePreapproachUnreachableError,'goal_cell_not_traversable'):
                    compute(.4)
                route = compute(.45)
                evidence = evaluate_route_uncertainty_admission(context.costmaps.base_costmap,
                    [p.pose for p in route.result.route.points],PlanarCovariance(**d['covariance']),
                    RouteUncertaintyAdmissionConfig(**d['uncertainty_config']))
                self.assertFalse(evidence.decision.accepted)
                self.assertAlmostEqual(evidence.decision.remaining_margin_m,d['recorded_margin_045_m'],places=6)
                # Merely refreshing while the envelope stays unchanged cannot
                # admit the route. Real improvement is needed; limits stay fixed.
                cfg = RouteUncertaintyAdmissionConfig(**d['uncertainty_config'])
                converged = evaluate_route_uncertainty_admission(context.costmaps.base_costmap,
                    [p.pose for p in route.result.route.points],PlanarCovariance(xx_m2=.002,xy_m2=0.,yy_m2=.002),
                    replace(cfg,heading_sigma_rad=.03))
                self.assertTrue(converged.decision.accepted)


if __name__ == '__main__':
    unittest.main()
