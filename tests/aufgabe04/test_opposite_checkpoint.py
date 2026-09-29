"""Recorded uncertainty regressions and fail-closed checkpoint handoff."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.approach.candidate_route_uncertainty_selection import CandidateRouteUncertaintyContext
from scripts.aufgabe04.navigation.approach.opposite_checkpoint_selection import select_opposite_checkpoint, OppositeCheckpoint
from scripts.aufgabe04.navigation.approach.opposite_checkpoint_route import materialize_opposite_checkpoint
from scripts.aufgabe04.navigation.approach.admitted_return_uncertainty import build_return_prefix
from scripts.aufgabe04.navigation.execution.route_uncertainty_admission import RouteUncertaintyAdmissionConfig, evaluate_route_uncertainty_admission
from scripts.aufgabe04.navigation.execution.route_uncertainty_budget import PlanarCovariance
from scripts.aufgabe04.navigation.execution.route_context import file_sha256
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.navigation.planning.map_io import load_occupancy_grid
from scripts.aufgabe04.navigation.planning.waypoint_csv import load_route_leg
from scripts.aufgabe04.real_robot.candidate import opposite_checkpoint as coordinator

REPO = Path(__file__).resolve().parents[2]


class RecordedCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((Path(__file__).parent/'fixtures/opposite_checkpoint_20260928.json').read_text())
        cls.costmap = Costmap.from_occupancy_grid(load_occupancy_grid(REPO/'maps/aufgabe03/arena_1p898x3p9_auto.yaml')).with_arena_bounds(ArenaBounds(**cls.record['arena_bounds']))

    def inputs(self, row):
        poses = tuple(Pose2D(p['x_m'],p['y_m'],math.nan if p['yaw_rad'] is None else p['yaw_rad']) for p in row['poses'])
        context = CandidateRouteUncertaintyContext(PlanarCovariance(**row['covariance']),RouteUncertaintyAdmissionConfig(**row['config']),{})
        return dict(poses=poses, start_pose=replace(poses[0],yaw_rad=0.), costmap=self.costmap,
                    uncertainty=context,source_sha256='a'*64)

    def test_all_seven_original_rejections_reproduced_without_relaxed_limits(self):
        self.assertEqual(len(self.record['routes']),7)
        for row in self.record['routes']:
            with self.subTest(row=row['attempt']):
                kw=self.inputs(row); u=kw['uncertainty']
                original=evaluate_route_uncertainty_admission(self.costmap,kw['poses'],u.covariance,u.admission_config)
                self.assertFalse(original.decision.accepted)
                self.assertAlmostEqual(original.decision.remaining_margin_m,row['margin_m'],places=9)
                choice=select_opposite_checkpoint(**kw)
                if choice:
                    self.assertGreater(choice.minimum_margin_m,0.)
                    self.assertFalse(choice.evidence['motion_authorized'])
                    self.assertTrue(choice.evidence['hypothetical_suffix_only'])
                self.assertEqual(kw['uncertainty'],u)

    def test_best_recorded_route_stops_at_expected_safe_vertex(self):
        row=next(r for r in self.record['routes'] if r['attempt']=='inspection_003_opposite_standoff_001_dry_uncertainty_budget.json')
        choice=select_opposite_checkpoint(**self.inputs(row))
        self.assertEqual(choice.vertex_index,3)
        self.assertAlmostEqual(choice.poses[-1].x_m,-1.545)
        self.assertAlmostEqual(choice.poses[-1].y_m,-.265)
        self.assertGreater(choice.minimum_margin_m,.045)
        self.assertLess(choice.minimum_margin_m,.048)

    def test_unhelpful_checkpoint_and_changed_anchor_do_not_pass(self):
        kw=self.inputs(self.record['routes'][0])
        with self.assertRaisesRegex(ValueError,'anchor'):
            select_opposite_checkpoint(**{**kw,'start_pose':Pose2D(0,0,0)})
        u=kw['uncertainty']
        impossible=replace(u,covariance=PlanarCovariance(xx_m2=1.,xy_m2=0.,yy_m2=1.))
        self.assertIsNone(select_opposite_checkpoint(**{**kw,'uncertainty':impossible}))
        self.assertIsNone(select_opposite_checkpoint(**{**kw,'poses':kw['poses'][:2]}))


class CheckpointArtifactTest(unittest.TestCase):
    def parent(self, root):
        from tests.aufgabe04.test_candidate_preapproach_planning import CandidatePreapproachPlanningTest, PHYSICAL_CLEARANCE
        from tests.aufgabe04.backside_axis_fixture import backside_axis_payload
        from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import plan_candidate_preapproach
        f=CandidatePreapproachPlanningTest()
        candidate,snapshot,snapshot_path,_=f._materialization_fixture(root)
        from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot
        candidate=replace(candidate,geometry=replace(candidate.geometry,x_m=0.))
        snapshot=replace(snapshot,candidates=(candidate,))
        snapshot_path=root/'checkpoint_source_snapshot.json'
        write_candidate_snapshot(snapshot_path,snapshot)
        axis=root/'axis.json'
        axis.write_text(json.dumps(backside_axis_payload(stand_x_m=0.,robot_x_m=-.6,robot_y_m=0.,stand_axis_rad=math.pi/2)))
        sealed=plan_candidate_preapproach(map_yaml=root/'map.yaml',semantic_map_id='arena',
            plan=f._plan(snapshot.map_bundle_sha256),snapshot=snapshot,snapshot_path=snapshot_path,
            candidate_uid=candidate.candidate_uid,start=Pose2D(-.6,0,0),output_dir=root/'parent',
            approach_offset_m=.45,inflation_radius_m=.25,candidate_transit_radius_m=.31,
            physical_clearance=PHYSICAL_CLEARANCE,approach_normal_rad=0.,axis_observation_path=axis)
        return sealed,snapshot_path

    def make(self,root):
        sealed,snapshot=self.parent(root)
        leg=load_route_leg(Path(sealed['route_csv']),0,thinning_min_spacing_m=0.)
        self.assertGreaterEqual(len(leg.raw_waypoints),3)
        poses=tuple(w.pose for w in leg.raw_waypoints)
        index=2
        prefix=build_return_prefix(poses,index-1,1.)
        choice=OppositeCheckpoint(index,prefix,.01,{'source_route_sha256':file_sha256(Path(sealed['route_csv']))})
        child=materialize_opposite_checkpoint(sealed_parent=sealed,snapshot_path=snapshot,choice=choice,output_dir=root/'checkpoint')
        return sealed,child,snapshot,choice

    def validate(self, child, snapshot):
        from scripts.aufgabe04.navigation.approach.detected_stand_preapproach import validate_detected_stand_preapproach_binding
        return validate_detected_stand_preapproach_binding(Path(child['diagnostics_json']),
            load_route_leg(Path(child['route_csv']),0,thinning_min_spacing_m=0.),candidate_snapshot_path=snapshot)

    def test_separate_prefix_seal_and_parent_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent,child,snapshot,choice=self.make(Path(tmp))
            self.assertTrue(self.validate(child,snapshot).ok)
            self.assertNotEqual(parent['route_certificate_json'],child['route_certificate_json'])
            metadata=json.loads(Path(child['diagnostics_json']).read_text())['metadata']
            self.assertFalse(metadata['opposite_localization_checkpoint']['camera_arrival'])
            self.assertEqual(metadata['opposite_localization_checkpoint']['vertex_index'],choice.vertex_index)
            # Neither a checkpoint chain nor a different parent can be substituted.
            Path(parent['diagnostics_json']).write_text('{}')
            self.assertFalse(self.validate(child,snapshot).ok)

    def test_metadata_and_prefix_tampering_rejected(self):
        for kind in ('center','index','yaw','candidate','prefix','nested'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                parent,child,snapshot,_=self.make(Path(tmp))
                p=Path(child['diagnostics_json']);d=json.loads(p.read_text());m=d['metadata']
                if kind=='center':m['validated_target_center']={'x_m':999}
                elif kind=='index':m['opposite_localization_checkpoint']['vertex_index']=0
                elif kind=='yaw':m['selected_approach_pose']['yaw_rad']+=.1
                elif kind=='candidate':m['selected_candidate_stand_id']='neighbor'
                elif kind=='prefix':
                    import csv
                    route=Path(child['route_csv'])
                    with route.open(newline='') as s:reader=csv.DictReader(s);fields=reader.fieldnames;rows=list(reader)
                    rows[-1]['world_x_m']=str(float(rows[-1]['world_x_m'])+.01)
                    with route.open('w',newline='') as s:writer=csv.DictWriter(s,fieldnames=fields);writer.writeheader();writer.writerows(rows)
                    m['route_csv_sha256']=file_sha256(route)
                else:
                    m['opposite_localization_checkpoint']['parent_diagnostics_json']=str(p)
                p.write_text(json.dumps(d))
                self.assertFalse(self.validate(child,snapshot).ok)


class CheckpointHandoffTest(unittest.TestCase):
    def exercise(self, prefix_error=None, suffix_error=None, prefix_status='completed', suffix_result='arrived'):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);poses=(Pose2D(0,0,0),Pose2D(.4,0,0),Pose2D(.8,0,0))
            choice=OppositeCheckpoint(1,poses[:2],.02,{'source_route_sha256':'a'*64})
            request=SimpleNamespace(output_dir=root/'parent')
            # A dataclass request is required by the production replace operation.
            from scripts.aufgabe04.real_robot.candidate.approach import CandidatePreapproachRequest
            request=CandidatePreapproachRequest(root/'map.yaml','arena',None,None,root/'snapshot','candidate',poses[0],root/'parent',.5,.25,.34,{})
            effects=SimpleNamespace(load_route_uncertainty_readiness=Mock(return_value='context'))
            config=SimpleNamespace(robot_radius_m=.105,planning_frame='map',uncertainty_sigma_multiplier=2.,map_yaml=root/'map.yaml',plan=SimpleNamespace(arena_bounds=None))
            order=[];events=[]
            def prefix(*args):
                order.append('prefix')
                if prefix_error:raise prefix_error
                return SimpleNamespace(status=prefix_status)
            def suffix():
                order.append('fresh_localization_then_suffix')
                if suffix_error:raise suffix_error
                return suffix_result
            with patch.object(coordinator,'load_occupancy_grid'),patch.object(coordinator.Costmap,'from_occupancy_grid'), \
                 patch.object(coordinator,'load_route_leg',return_value=SimpleNamespace(raw_waypoints=[SimpleNamespace(pose=p) for p in poses])), \
                 patch.object(coordinator,'file_sha256',return_value='a'*64),patch.object(coordinator,'select_opposite_checkpoint',return_value=choice), \
                 patch.object(coordinator,'materialize_opposite_checkpoint',return_value={'route_csv':'prefix'}) as seal:
                kwargs=dict(rejected_routes=[(request,{'route_csv':'parent'},'run')],config=config,effects=effects,
                    planning_frame=SimpleNamespace(current_pose=poses[0],odom_frame='odom'),candidate_root=root,
                    execute_prefix=prefix,continue_from_checkpoint=suffix,event_sink=events.append)
                if prefix_error or suffix_error or prefix_status != 'completed' or suffix_result is None:
                    with self.assertRaises(coordinator.OppositeCheckpointExecutionError):coordinator.try_opposite_checkpoint(**kwargs)
                else:self.assertEqual(coordinator.try_opposite_checkpoint(**kwargs),'arrived')
                seal.assert_called_once()
                self.assertEqual(order,['prefix'] if prefix_error or prefix_status != 'completed' else ['prefix','fresh_localization_then_suffix'])
                self.assertEqual(events[0]['maximum_checkpoint_count'],1)

    def test_prefix_then_fresh_suffix_once(self):self.exercise()
    def test_failed_prefix_never_dispatches_suffix(self):self.exercise(prefix_error=RuntimeError('motion failed'))
    def test_incomplete_prefix_never_dispatches_suffix(self):self.exercise(prefix_status='blocked')
    def test_missing_arrival_cannot_fall_through_to_no_motion_retry(self):self.exercise(suffix_result=None)
    def test_suffix_route_unavailable_cannot_escape_as_no_motion_failure(self):
        from scripts.aufgabe04.real_robot.candidate.inspection_route_search import CandidateInspectionRouteUnavailableError
        self.exercise(suffix_error=CandidateInspectionRouteUnavailableError('fresh budget rejected'))


class OppositeCoordinatorCheckpointTest(unittest.TestCase):
    def test_only_refreshed_exhaustion_uses_checkpoint_then_reprojects_original_receipt(self):
        from scripts.aufgabe04.real_robot.candidate import approach
        from tests.aufgabe04.test_autonomous_candidate_approach import AutonomousCandidateApproachTest
        from tests.aufgabe04.test_opposite_face_route_fallback import OppositeFaceRouteFallbackTest
        factory=AutonomousCandidateApproachTest()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            candidate=factory._candidate('candidate_1',.2,0.)
            config=replace(factory._config(root,(candidate,)),approach_offset_m=.5)
            frame=approach._CandidateObservationFrame(config,candidate,None,None)
            observation=approach.CandidateObservation(None,None,root/'original_axis.json')
            admissions=[];plans=[];motions=[];checkpoints=[]
            def admit(**kw):
                admissions.append(kw['candidate_root'])
                return config,candidate,Pose2D(.1*len(admissions),0,0),None,None
            def plan(request):
                plans.append(request)
                return {'route_csv':str(request.output_dir/'route.csv')}
            def motion(**kw):
                motions.append(kw)
                if '_checkpoint' in kw['run_id']:
                    return SimpleNamespace(status='completed')
                outcome=factory._route_uncertainty_rejection(SimpleNamespace(run_id=kw['run_id'],session_root=root))
                raise OppositeFaceRouteFallbackTest._error(outcome)
            def checkpoint(**kw):
                checkpoints.append(kw)
                self.assertEqual(len(admissions),2)
                request,sealed,run_id=kw['rejected_routes'][-1]
                kw['execute_prefix'](request,sealed,run_id+'_checkpoint')
                return kw['continue_from_checkpoint']()
            effects=approach.CandidateApproachEffects(read_current_pose=Mock(),run_motion_leg=Mock(),
                capture_observation=Mock(),plan_preapproach=plan,admit_planning_frame=Mock(),event_sink=Mock())
            arrived=object()
            with patch.object(approach,'_admit_opposite_face_planning_geometry',side_effect=admit), \
                 patch.object(approach,'opposite_face_normal',return_value=1.2), \
                 patch.object(approach,'load_backside_axis_planning_observation',return_value=SimpleNamespace(validated_target_center={'uncertainty_m':.02})), \
                 patch.object(approach,'bounded_inspection_standoffs',return_value=(.5,.45)), \
                 patch.object(approach,'_execute_candidate_motion',side_effect=motion), \
                 patch.object(approach,'try_opposite_checkpoint',side_effect=checkpoint), \
                 patch.object(approach,'_admit_camera_arrival_geometry',return_value=arrived):
                result=approach._move_certified_opposite_face(observation_frame=frame,observation=observation,
                    source_config=config,effects=effects,source_registry=None,candidate_root=root/'opposite',
                    candidate_run_id='mission_inspect',candidate_index=1)
            self.assertIs(result,arrived)
            self.assertEqual(len(checkpoints),1)
            self.assertEqual(len(admissions),3)
            self.assertEqual(admissions[-1],root/'opposite/localization_001/after_checkpoint')
            self.assertTrue(all(p.axis_observation_path==observation.axis_observation_path for p in plans))
            self.assertTrue(all(p.approach_normal_rad==1.2 for p in plans))
            self.assertEqual(plans[-1].start,Pose2D(.1*3,0,0))
            prefix=next(m for m in motions if m['run_id'].endswith('_checkpoint'))
            self.assertEqual(prefix['config'].max_startup_reseals_per_leg,0)
            self.assertEqual(prefix['config'].max_runtime_localization_reseals_per_leg,0)
            self.assertEqual(len({m['run_id'] for m in motions}),len(motions))


if __name__=='__main__':unittest.main()
