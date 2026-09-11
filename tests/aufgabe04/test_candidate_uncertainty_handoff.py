"""Recorded regression at the real planning/uncertainty consumer boundary."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosObservation, RosPreflightResult,
)
from scripts.aufgabe04.real_robot.readiness.candidate_planning_frame import (
    build_candidate_planning_frame,
)
from scripts.aufgabe04.real_robot.candidate.route_uncertainty_readiness import (
    CandidateRouteUncertaintyReadinessRequest,
    load_candidate_route_uncertainty_readiness,
)
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, execute_candidate_approach_phase,
)
from tests.aufgabe04 import test_autonomous_candidate_approach as approach_fixtures


FIXTURE = Path(__file__).parent / "fixtures/candidate_planning_frame_20260911T133125Z.json"


def recorded_fixture():
    return json.loads(FIXTURE.read_text())


def planning_frame(payload):
    preflight = RosPreflightResult(
        **{key: value for key, value in payload.items() if key != "observations"},
        observations=[RosObservation(**value) for value in payload["observations"]],
    )
    chained = Pose2D(**{key: payload["route_pose"][key] for key in ("x_m", "y_m", "yaw_rad")})
    return build_candidate_planning_frame(
        preflight, current_pose=chained,
        map_frame=payload["runtime_config"]["map_frame"],
        odom_frame=payload["runtime_config"]["odom_frame"],
    )


class _SelectorReached(Exception):
    pass


class CandidateUncertaintyHandoffTest(unittest.TestCase):
    def test_recorded_frame_reaches_real_uncertainty_adapter(self):
        fixture = recorded_fixture()
        payload = fixture["preflight"]
        frame = planning_frame(payload)
        self.assertEqual(frame.to_evidence(), fixture["expected_planning_frame"])
        self.assertAlmostEqual(
            frame.pose_provenance["chained_to_authoritative_translation_delta_m"],
            0.004112032257420838,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "preflight.json"
            source.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            original = source.read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), fixture["preflight_sha256"])
            request = CandidateRouteUncertaintyReadinessRequest(
                preflight_json=source, expected_start=frame.current_pose,
                planning_frame=frame.map_frame, odom_frame=frame.odom_frame,
                robot_radius_m=0.105, sigma_multiplier=2.0,
            )
            context = load_candidate_route_uncertainty_readiness(request)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(context.source_evidence["source_preplanning_localization_sha256"], fixture["preflight_sha256"])
            self.assertEqual(context.source_evidence["admitted_start_pose"], fixture["expected_planning_frame"]["current_pose"])
            self.assertEqual(context.source_evidence["pose_provenance"]["map_from_odom_capture"], payload["map_from_odom"])
            self.assertEqual(context.admission_config.fixed_odom_tracking_bound_m, 0.03)
            self.assertFalse(context.source_evidence["motion_authorized"])
            # Neither the diagnostic chained pose nor even a tiny unrelated
            # start change satisfies the candidate identity contract.
            chained = Pose2D(**{key: payload["route_pose"][key] for key in ("x_m", "y_m", "yaw_rad")})
            for wrong in (chained, replace(frame.current_pose, x_m=frame.current_pose.x_m + 1e-10)):
                with self.subTest(start=wrong), self.assertRaisesRegex(ValueError, "does not match"):
                    load_candidate_route_uncertainty_readiness(replace(request, expected_start=wrong))

    def test_coordinator_reaches_selector_with_real_admission_and_adapter(self):
        self._coordinator_case()

    def test_coordinator_passes_explicit_nondefault_odom_frame(self):
        self._coordinator_case(odom_frame="robot_1/odom")

    def test_changed_capture_rejected_before_selector_or_motion(self):
        self._coordinator_case(tamper=True)

    def _coordinator_case(self, *, odom_frame="odom", tamper=False):
        payload = deepcopy(recorded_fixture()["preflight"])
        if odom_frame != "odom":
            def rename(value):
                if isinstance(value, dict):
                    return {k: rename(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [rename(v) for v in value]
                if isinstance(value, str):
                    return {
                        "odom": odom_frame,
                        "tf map->odom": f"tf map->{odom_frame}",
                        "tf odom->base_footprint": f"tf {odom_frame}->base_footprint",
                    }.get(value, value)
                return value
            payload = rename(payload)
        frame = planning_frame(payload)
        factory = approach_fixtures.AutonomousCandidateApproachTest()
        with tempfile.TemporaryDirectory() as tmp:
            config = factory._config(Path(tmp), (factory._candidate("candidate_a", 0.5, 0.5),))
            config = factory._write_frame_registry(config, frozen_map_from_odom=frame.map_from_odom)
            # The existing fixture's canonical candidate provenance uses odom;
            # update its frame label for the namespaced-frame plumbing case.
            if odom_frame != "odom":
                registry = config.survey_root / "stand_registry.json"
                # Reprojection and admission must name the same odom frame.
                from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
                    load_stand_survey_registry, write_stand_survey_registry,
                    stand_survey_registry_sha256,
                )
                from scripts.aufgabe04.stations.candidate_snapshot import write_candidate_snapshot
                saved = load_stand_survey_registry(registry, config.plan)
                saved = replace(saved, candidates=tuple(replace(c, frame_provenance=replace(c.frame_provenance, odom_frame=odom_frame)) for c in saved.candidates))
                write_stand_survey_registry(registry, saved, config.plan)
                digest = stand_survey_registry_sha256(saved)
                snapshot = replace(config.snapshot, candidates=tuple(replace(c, source=replace(c.source, source_artifact_sha256=digest)) for c in config.snapshot.candidates))
                snapshot_path = Path(tmp) / "namespaced_candidate_snapshot.json"
                write_candidate_snapshot(snapshot_path, snapshot)
                config = replace(config, snapshot=snapshot, snapshot_path=snapshot_path)
            config = replace(config, require_uncertainty_aware_selection=True, robot_radius_m=0.105)

            def admit(path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                admitted = planning_frame(payload)
                if tamper:
                    changed = deepcopy(payload)
                    changed["odom_pose"]["x_m"] += 0.01
                    path.write_text(json.dumps(changed) + "\n")
                return admitted

            selector = Mock(side_effect=_SelectorReached)
            motion = Mock(side_effect=AssertionError("motion must not run"))
            camera = Mock(side_effect=AssertionError("camera must not run"))
            old_pose_reader = Mock(side_effect=AssertionError("unadmitted pose must not run"))
            loader = Mock(wraps=load_candidate_route_uncertainty_readiness)
            effects = CandidateApproachEffects(
                read_current_pose=old_pose_reader,
                run_motion_leg=motion, capture_observation=camera,
                admit_planning_frame=admit, load_route_uncertainty_readiness=loader,
                select_initial_preapproach=selector,
            )
            with self.assertRaises(ValueError if tamper else _SelectorReached):
                execute_candidate_approach_phase(config, effects)
            loader.assert_called_once()
            self.assertEqual(loader.call_args.args[0].odom_frame, odom_frame)
            motion.assert_not_called()
            camera.assert_not_called()
            old_pose_reader.assert_not_called()
            if tamper:
                selector.assert_not_called()
                return
            selector.assert_called_once()
            request = selector.call_args.args[0]
            self.assertEqual(request.current_pose, frame.current_pose)
            self.assertEqual(request.route_uncertainty_context.source_evidence["admitted_start_pose"], frame.to_evidence()["current_pose"])
            self.assertEqual(request.route_uncertainty_context.source_evidence["pose_basis"], "direct_map_from_odom_times_observed_odom_pose")


if __name__ == "__main__":
    unittest.main()
