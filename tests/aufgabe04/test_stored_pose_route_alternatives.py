"""Bounded tour search preserves the physical floor and immutable evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.approach import admitted_pose_route as planner
from scripts.aufgabe04.navigation.approach import stored_pose_route_alternatives as alternatives
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import ReturnUncertaintyExhausted
from scripts.aufgabe04.navigation.execution.execution_route_certificate import file_sha256
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from tests.aufgabe04.test_admitted_return_uncertainty import _context
from tests.aufgabe04 import test_temporary_obstacle_route as route_fixtures
from tests.aufgabe04.test_stored_pose_tour_authorization import catalog_evidence


class RouteAlternativesPolicyTest(unittest.TestCase):
    def arguments(self):
        start = Pose2D(0., 0., 0.)
        return dict(base_radius_m=.25, base_costmap=SimpleNamespace(resolution=.05),
            uncertainty=_context(start), target_evidence_sha256='a'*64, identity={'tour_id':'tour'})

    def geometry(self, x=1.):
        return alternatives.StoredPoseRouteGeometry(None, None, SimpleNamespace(required=False), None,
            (Pose2D(0.,0.,0.), Pose2D(x,0.,.1)))

    def test_five_ascending_half_cell_radii_and_invalid_inputs(self):
        self.assertEqual(alternatives.alternative_inflation_radii(.25,.05), (.25,.275,.30,.325,.35))
        for floor, resolution in ((True,.05),(.25,0.),(.25,float('nan')),(-1.,.05)):
            with self.assertRaises(ValueError):
                alternatives.alternative_inflation_radii(floor,resolution)

    def test_same_context_first_passing_and_duplicate_rejection_cache(self):
        seen=[]; args=self.arguments(); geometry=self.geometry()
        stage=SimpleNamespace(evidence={'selected_admission':{'decision':'test'}},poses=geometry.full_poses,
                              is_final_stage=True,stage_target_pose=geometry.full_poses[-1])
        def build(radius):
            seen.append(radius)
            return geometry if radius < .325 else self.geometry(2.)
        def evaluate(**kwargs):
            self.assertIs(kwargs['base_costmap'],args['base_costmap'])
            self.assertIs(kwargs['uncertainty'],args['uncertainty'])
            if kwargs['full_poses'][-1].x_m==1.:
                raise ReturnUncertaintyExhausted('rejected',[{'remaining_margin_m':-.01}])
            return stage
        with patch.object(alternatives,'select_admitted_return_prefix',side_effect=evaluate) as evaluator:
            _,_,radius,evidence=alternatives.select_stored_pose_route_alternative(build,**args)
        self.assertEqual(seen,[.25,.275,.30,.325]);self.assertEqual(radius,.325)
        self.assertEqual(evaluator.call_count,2)
        self.assertEqual(evidence['attempts'][2]['uncertainty_evaluation_reused_from_attempt'],0)
        self.assertFalse(evidence['motion_authorized'])
        self.assertEqual(evidence['uncertainty_context_sha256'],payload_sha256(alternatives.uncertainty_context_evidence(args['uncertainty'])))

    def test_only_typed_geometry_or_uncertainty_rejections_retry(self):
        calls=[]
        def invalid(radius):
            calls.append(radius);raise ValueError('source hash mismatch')
        with self.assertRaisesRegex(ValueError,'source hash mismatch'):
            alternatives.select_stored_pose_route_alternative(invalid,**self.arguments())
        self.assertEqual(calls,[.25])
        def rejected(radius): raise alternatives.AlternativeGeometryRejected('exact target blocked')
        with self.assertRaises(alternatives.RouteAlternativesExhausted) as caught:
            alternatives.select_stored_pose_route_alternative(rejected,**self.arguments())
        self.assertEqual(len(caught.exception.evidence['attempts']),5)
        self.assertIsNone(caught.exception.evidence['selected_attempt_index'])


class RouteAlternativesBindingTest(unittest.TestCase):
    def fixture(self,root):
        return route_fixtures.TemporaryObstacleRouteTest().fixture(root,dynamic=True)[0]

    def test_dynamic_route_binds_actual_radius_and_search_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));result=planner.plan_admitted_pose_route(**args)
            metadata=json.loads(Path(result['diagnostics_json']).read_text())['metadata']
            evidence=json.loads(Path(metadata['route_alternatives_json']).read_text())
            self.assertEqual(evidence['selected_attempt_index'],0)
            self.assertEqual(metadata['inflation_radius_m'],args['inflation_radius_m'])
            self.assertEqual(evidence['robot_radius_m'],.105)
            self.assertEqual(evidence['attempts'][0]['full_route_geometry']['poses'][-1],asdict(args['target']))
            leg=load_route_leg(Path(result['route_csv']),0,thinning_min_spacing_m=0.)
            for key,bad in (('route_alternatives_sha256','0'*64),('inflation_radius_m',.20)):
                altered={'metadata':{**metadata,key:bad}}
                status=planner.validate_admitted_pose_route_binding(Path(result['diagnostics_json']),leg,
                    candidate_snapshot_path=Path(result['candidate_snapshot']),diagnostics_payload=altered)
                self.assertFalse(status.ok,status)
            # Even editing the artifact and refreshing both enclosing hashes
            # cannot claim a different cell schedule or omit the selected proof.
            artifact=Path(metadata['route_alternatives_json'])
            full_path=Path(metadata['return_to_start_stage']['full_return_route_json'])
            full=json.loads(full_path.read_text())
            original=artifact.read_text()
            for change in ({'map_resolution_m':.1},{'base_inflation_radius_m':.20},{'selected_attempt_index':1}):
                artifact.write_text(json.dumps({**evidence,**change}))
                altered=deepcopy(metadata);altered['route_alternatives_sha256']=file_sha256(artifact)
                full_path.write_text(json.dumps({**full,'route_alternatives_sha256':file_sha256(artifact)}))
                altered['return_to_start_stage']['full_return_route_sha256']=file_sha256(full_path)
                status=planner.validate_admitted_pose_route_binding(Path(result['diagnostics_json']),leg,
                    candidate_snapshot_path=Path(result['candidate_snapshot']),diagnostics_payload={'metadata':altered})
                self.assertFalse(status.ok,status)
            artifact.write_text(original)

    def test_dynamic_tour_cannot_strip_both_alternative_artifact_references(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));result=planner.plan_admitted_pose_route(**args)
            diagnostics=json.loads(Path(result['diagnostics_json']).read_text())
            metadata=diagnostics['metadata']
            self.assertIn('tour_navigation',metadata)
            full_path=Path(metadata['return_to_start_stage']['full_return_route_json'])
            full=json.loads(full_path.read_text())
            for payload in (metadata,full):
                payload.pop('route_alternatives_json')
                payload.pop('route_alternatives_sha256')
            full_path.write_text(json.dumps(full))
            metadata['return_to_start_stage']['full_return_route_sha256']=file_sha256(full_path)
            leg=load_route_leg(Path(result['route_csv']),0,thinning_min_spacing_m=0.)
            status=planner.validate_admitted_pose_route_binding(Path(result['diagnostics_json']),leg,
                candidate_snapshot_path=Path(result['candidate_snapshot']),diagnostics_payload=diagnostics)
            self.assertFalse(status.ok,status)
            self.assertIn('route_alternatives_json',status.failures[0])

    def test_exhaustion_saves_geometry_and_budget_without_route_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory))
            context=args['route_uncertainty_context']
            args['route_uncertainty_context']=replace(context,covariance=replace(context.covariance,xx_m2=2.,yy_m2=2.))
            with self.assertRaisesRegex(ValueError,'bounded route alternatives'):
                planner.plan_admitted_pose_route(**args)
            self.assertFalse(args['output_dir'].exists())
            failure=args['output_dir'].with_name(args['output_dir'].name+'_alternatives_failure.json')
            evidence=json.loads(failure.read_text())
            self.assertEqual(len(evidence['attempts']),5)
            first=evidence['attempts'][0]
            self.assertEqual(first['full_route_geometry']['poses'][-1],asdict(args['target']))
            self.assertLess(first['admission_attempts'][0]['remaining_margin_m'],0.)
            self.assertFalse(evidence['motion_authorized'])
            self.assertFalse(list(Path(directory).rglob('route_certificate.json')))

    def test_stationary_dynamic_tour_keeps_existing_single_point_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root)
            target=replace(args['start'],yaw_rad=args['start'].yaw_rad+.5)
            args['target']=target
            args['target_evidence'].update(catalog_evidence(root/'stationary-source',uid=args['candidate_uid'],pose=asdict(target)))
            args['target_evidence']['target_pose']=asdict(target)
            with patch.object(planner,'select_stored_pose_route_alternative') as search:
                result=planner.plan_admitted_pose_route(**args)
                search.assert_not_called()
            metadata=json.loads(Path(result['diagnostics_json']).read_text())['metadata']
            self.assertTrue(metadata['stationary_turn'])
            self.assertNotIn('route_alternatives_json',metadata)
            self.assertEqual(result['stage_target_pose'],asdict(target))

    def test_final_stage_limit_is_not_converted_into_another_geometry_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));args['return_stage_index']=3
            args['target_evidence']['tour_navigation'].update(stage_index=3,execution_index=3,
                previous_terminal_json='/tmp/previous-terminal.json',previous_terminal_sha256='a'*64)
            original=planner.select_stored_pose_route_alternative
            def force_nonfinal(*a,**kw):
                geometry,stage,radius,evidence=original(*a,**kw)
                return geometry,replace(stage,is_final_stage=False),radius,evidence
            with patch.object(planner,'select_stored_pose_route_alternative',side_effect=force_nonfinal) as search:
                with self.assertRaisesRegex(ValueError,'stage limit exhausted'):
                    planner.plan_admitted_pose_route(**args)
                self.assertEqual(search.call_count,1)
            self.assertFalse(args['output_dir'].exists())

    def test_bad_source_and_measured_center_never_enter_search(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.fixture(Path(directory))
            with patch.object(planner,'select_stored_pose_route_alternative') as search:
                args['target_evidence']['measured_target_center']={'x_m':float('nan'),'y_m':0.,'uncertainty_m':0.}
                with self.assertRaises(ValueError):planner.plan_admitted_pose_route(**args)
                search.assert_not_called()
                args['target_evidence'].pop('measured_target_center')
                source=Path(args['target_evidence']['source_artifacts'][0]['path']);source.write_text('changed')
                with self.assertRaisesRegex(ValueError,'hash'):planner.plan_admitted_pose_route(**args)
                search.assert_not_called()
            self.assertFalse(args['output_dir'].exists())
