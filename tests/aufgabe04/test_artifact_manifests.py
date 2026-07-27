import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.aufgabe04.artifacts import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactManifestError,
    ExecutionEvidenceManifest,
    MissionPlanManifest,
    SurveyManifest,
    artifact_reference,
    execution_evidence_manifest_sha256,
    load_execution_evidence_manifest,
    load_mission_plan_manifest,
    load_survey_manifest,
    manifest_reference,
    mission_plan_manifest_sha256,
    survey_manifest_sha256,
    write_execution_evidence_manifest,
    write_mission_plan_manifest,
    write_survey_manifest,
)


def _ref(kind, artifact_id, marker):
    return artifact_reference(kind, artifact_id, marker * 64)


def _survey():
    return SurveyManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        manifest_id="survey_manifest_001",
        created_unix_sec=100.0,
        session_id="gazebo_session_001",
        environment="simulation",
        planning_frame="map",
        map_bundle=_ref("map_bundle", "temporary_map_001", "a"),
        candidate_snapshot=_ref("candidate_snapshot", "candidates_001", "b"),
        environment_descriptor=_ref("simulation_world", "world_001", "c"),
        survey_config=_ref("survey_config", "survey_config_001", "d"),
        calibration_profile=_ref("calibration_profile", "sim_camera_001", "e"),
        arrival_pose_catalog=_ref("arrival_pose_catalog", "catalog_001", "f"),
    )


def _mission(survey=None):
    selected = survey or _survey()
    return MissionPlanManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        manifest_id="mission_plan_001",
        created_unix_sec=110.0,
        robot_id="tb3_1",
        parent_survey_manifest=manifest_reference(
            selected, survey_manifest_sha256(selected)
        ),
        map_bundle=selected.map_bundle,
        candidate_snapshot=selected.candidate_snapshot,
        station_identity_registry=_ref(
            "station_identity_registry", "identities_001", "1"
        ),
        arrival_pose_catalog=selected.arrival_pose_catalog,
        task_snapshot=_ref("task_snapshot", "server_task_001", "2"),
        planner_config=_ref("planner_config", "planner_config_001", "3"),
        route_bundle=_ref("route_bundle", "route_001", "4"),
        required_station_order=("station_C", "station_A", "station_B"),
        ordered_candidate_uids=("candidate_c", "candidate_a", "candidate_b"),
    )


def _execution(mission=None):
    selected = mission or _mission()
    return ExecutionEvidenceManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        manifest_id="execution_evidence_001",
        created_unix_sec=130.0,
        attempt_id="attempt_001",
        robot_id="tb3_1",
        parent_mission_plan_manifest=manifest_reference(
            selected, mission_plan_manifest_sha256(selected)
        ),
        controller_profile=_ref("controller_profile", "controller_001", "5"),
        route_certificate=_ref("route_certificate", "certificate_001", "6"),
        event_log=_ref("event_log", "events_001", "7"),
        execution_summary=_ref("execution_summary", "summary_001", "8"),
        started_unix_sec=120.0,
        finished_unix_sec=129.0,
        outcome="completed",
    )


class ArtifactManifestTest(unittest.TestCase):
    def test_all_manifest_types_round_trip_with_verified_hashes(self):
        survey = _survey()
        mission = _mission(survey)
        execution = _execution(mission)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            survey_hash = write_survey_manifest(root / "survey.json", survey)
            mission_hash = write_mission_plan_manifest(root / "mission.json", mission)
            execution_hash = write_execution_evidence_manifest(
                root / "execution.json", execution
            )
            loaded = (
                load_survey_manifest(root / "survey.json"),
                load_mission_plan_manifest(
                    root / "mission.json", parent_survey=survey
                ),
                load_execution_evidence_manifest(
                    root / "execution.json", parent_mission=mission
                ),
            )

        self.assertEqual(loaded, (survey, mission, execution))
        self.assertEqual(survey_hash, survey_manifest_sha256(survey))
        self.assertEqual(mission_hash, mission_plan_manifest_sha256(mission))
        self.assertEqual(
            execution_hash, execution_evidence_manifest_sha256(execution)
        )

    def test_parent_hash_changes_cascade_through_lifecycle(self):
        survey = _survey()
        changed_survey = replace(
            survey, survey_config=_ref("survey_config", "survey_config_001", "9")
        )
        mission = _mission(survey)
        changed_mission = replace(
            mission,
            parent_survey_manifest=manifest_reference(
                changed_survey, survey_manifest_sha256(changed_survey)
            ),
        )
        execution = _execution(mission)
        changed_execution = replace(
            execution,
            parent_mission_plan_manifest=manifest_reference(
                changed_mission, mission_plan_manifest_sha256(changed_mission)
            ),
        )

        self.assertNotEqual(
            survey_manifest_sha256(survey),
            survey_manifest_sha256(changed_survey),
        )
        self.assertNotEqual(
            mission_plan_manifest_sha256(mission),
            mission_plan_manifest_sha256(changed_mission),
        )
        self.assertNotEqual(
            execution_evidence_manifest_sha256(execution),
            execution_evidence_manifest_sha256(changed_execution),
        )

    def test_immutable_publish_rejects_different_manifest(self):
        survey = _survey()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "survey.json"
            first = write_survey_manifest(path, survey)
            retry = write_survey_manifest(path, survey)
            with self.assertRaises(ArtifactManifestError) as raised:
                write_survey_manifest(
                    path, replace(survey, manifest_id="survey_manifest_002")
                )

        self.assertEqual(first, retry)
        self.assertEqual(raised.exception.code, "immutable_conflict")

    def test_tampering_and_unknown_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "survey.json"
            write_survey_manifest(path, _survey())
            payload = json.loads(path.read_text())
            payload["session_id"] = "tampered"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ArtifactManifestError) as tampered:
                load_survey_manifest(path)
        self.assertEqual(tampered.exception.code, "hash_mismatch")

    def test_wrong_link_type_order_lengths_and_execution_times_are_rejected(self):
        with self.assertRaises(ArtifactManifestError):
            survey_manifest_sha256(
                replace(
                    _survey(),
                    map_bundle=_ref("candidate_snapshot", "wrong", "a"),
                )
            )
        with self.assertRaises(ArtifactManifestError):
            mission_plan_manifest_sha256(
                replace(_mission(), ordered_candidate_uids=("candidate_a",))
            )
        with self.assertRaises(ArtifactManifestError):
            execution_evidence_manifest_sha256(
                replace(_execution(), finished_unix_sec=119.0)
            )

    def test_parent_link_and_duplicated_provenance_must_match(self):
        survey = _survey()
        mission = replace(
            _mission(survey), map_bundle=_ref("map_bundle", "other_map", "9")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_mission_plan_manifest(root / "mission.json", mission)
            with self.assertRaises(ArtifactManifestError) as mission_mismatch:
                load_mission_plan_manifest(
                    root / "mission.json", parent_survey=survey
                )

            execution = replace(_execution(), robot_id="tb3_2")
            write_execution_evidence_manifest(root / "execution.json", execution)
            with self.assertRaises(ArtifactManifestError) as robot_mismatch:
                load_execution_evidence_manifest(
                    root / "execution.json", parent_mission=_mission()
                )

        self.assertEqual(mission_mismatch.exception.code, "provenance_mismatch")
        self.assertEqual(robot_mismatch.exception.code, "provenance_mismatch")


if __name__ == "__main__":
    unittest.main()
