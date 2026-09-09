"""Regression scenarios for distinct QR goals over a larger hypothesis pool."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.candidate_perception_advisory import (
    CandidatePerceptionAdvisory, VISIBILITY_GAP,
)
from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, write_content_hashed_json,
)
from scripts.aufgabe04.navigation.approach.camera_candidate_selection import NoFeasibleCameraCandidateError
from scripts.aufgabe04.navigation.approach.candidate_preapproach_planning import (
    CandidatePreapproachUnreachableError,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance, CandidatePoint2D,
)
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import PlanarTransform2D
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.real_robot.candidate.approach import (
    CandidateApproachEffects, CandidateObservation, execute_candidate_approach_phase,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.qr_goal_progress import (
    GOAL_PROGRESS_HASH_FIELD, CandidateQrGoalIncompleteError,
    CandidateQrGoalProgress, validate_candidate_qr_goal_completion,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_snapshot_sha256, load_candidate_snapshot,
)
from scripts.aufgabe04.stations.server_identity_binding import load_observed_identities
from tests.aufgabe04 import test_autonomous_candidate_approach as fixtures


class CandidateQrGoalExecutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = fixtures.AutonomousCandidateApproachTest()
        self.visited = []
        self.planned_snapshots = []
        self.committed = []

    def config(self, pool_size=6):
        config = self.fixture._config(self.root, tuple(
            self.fixture._candidate(f"candidate_{index}", 0.2 + index * 0.1, 0.0)
            for index in range(pool_size)
        ))
        return replace(config, expected_stand_count=5, max_candidate_inspection_views=1,
                       plan=replace(config.plan, config=replace(config.plan.config, expected_stand_count=5)))

    def effects(self, *, exhausted=(), qr_by_uid=None, unreachable=(), failure=None):
        def capture(request):
            uid = request.candidate.candidate_uid
            self.visited.append(uid)
            if failure == "tf":
                raise RuntimeError("terminal TF discontinuity")
            if uid in exhausted:
                raise CandidateObservationUnavailableError(
                    candidate_uid=uid, observation_attempt_index=0,
                    reason="no jointly valid QR and axis", process_evidence={}, status_evidence={},
                )
            return CandidateObservation(
                request.output_dir / "recommendation.json",
                (qr_by_uid or {}).get(uid, f"QR_{uid}"), None,
            )

        def plan(request):
            self.planned_snapshots.append(request.snapshot)
            if failure == "certificate":
                raise ValueError("terminal route certificate hash mismatch")
            if request.candidate_uid in unreachable:
                raise CandidatePreapproachUnreachableError(request.candidate_uid, "static route blocked")
            return {"route_csv": "route.csv"}

        def motion(request):
            if failure == "motion":
                raise RuntimeError("terminal child motion failure")
            return self.fixture._completed(request)

        def commit(request):
            self.committed.append(json.loads(request.receipt_path.read_text())["candidate_uid"])

        return CandidateApproachEffects(
            select_initial_preapproach=self.fixture._nearest_selection,
            read_current_pose=lambda: Pose2D(0.0, 0.0, 0.0),
            plan_preapproach=plan, run_motion_leg=motion, capture_observation=capture,
            validate_facing=lambda request: {"candidate_uid": request.candidate.candidate_uid},
            commit_decision=commit, clock=lambda: 10.0,
        )

    def assert_bound_artifacts(self, config, result):
        confirmed = load_candidate_snapshot(result.confirmed_candidate_snapshot_path)
        self.assertIsNone(result.identity_registry_path)
        self.assertEqual(result.identity_binding_status, "server_binding_pending")
        identity = load_observed_identities(result.observed_identities_path,
                                            candidate_snapshot=confirmed)["observed_qr_by_candidate"]
        progress = validate_candidate_qr_goal_completion(
            result.candidate_goal_progress_path, candidate_snapshot=config.snapshot,
            confirmed_candidate_snapshot=confirmed, observed_qr_by_candidate=identity,
            expected_stand_count=5,
        )
        self.assertEqual(result.stand_count, 5)
        self.assertEqual(len(identity), 5)
        self.assertEqual(len(set(identity.values())), 5)
        self.assertTrue(all(snapshot == config.snapshot for snapshot in self.planned_snapshots))
        self.assertEqual(progress["keepout_candidate_uids"], list(config.snapshot.candidate_uids))
        self.assertEqual(progress["candidate_snapshot_sha256"], candidate_snapshot_sha256(config.snapshot))
        self.assertTrue(progress["goal_completed"])
        return progress, confirmed, identity

    def test_six_hypotheses_one_exhausted_then_five_identities_complete(self):
        config = self.config()
        result = execute_candidate_approach_phase(config, self.effects(exhausted={"candidate_0"}))
        progress, _, _ = self.assert_bound_artifacts(config, result)
        self.assertEqual(self.visited, [f"candidate_{i}" for i in range(6)])
        dispositions = {item["candidate_uid"]: item["disposition"] for item in progress["candidate_dispositions"]}
        self.assertEqual(dispositions["candidate_0"], "inspection_exhausted")
        self.assertNotIn("candidate_0", result.visit_order)
        self.assertEqual(progress["remaining_candidate_uids"], ["candidate_0"])

    def test_goal_stops_before_sixth_candidate_and_retains_its_keepout(self):
        config = self.config()
        result = execute_candidate_approach_phase(config, self.effects())
        progress, _, _ = self.assert_bound_artifacts(config, result)
        self.assertEqual(self.visited, [f"candidate_{i}" for i in range(5)])
        self.assertEqual(progress["unvisited_candidate_uids"], ["candidate_5"])
        self.assertEqual(progress["candidate_dispositions"][-1]["disposition"], "not_visited_goal_reached")
        self.assertEqual(result.to_mission_summary_fields()["candidate_pool_count"], 6)
        self.assertNotIn("schema_version", result.to_mission_summary_fields())

    def test_duplicate_qr_quarantines_both_claimants_and_does_not_count_twice(self):
        config = self.config(pool_size=7)
        result = execute_candidate_approach_phase(config, self.effects(
            qr_by_uid={"candidate_0": "QR_DUPLICATE", "candidate_1": "QR_DUPLICATE"},
        ))
        progress, _, identity = self.assert_bound_artifacts(config, result)
        self.assertEqual(len(self.visited), 7)
        self.assertEqual(len(result.visit_order), 5)
        self.assertNotIn("QR_DUPLICATE", list(identity.values()))
        self.assertNotIn("candidate_1", self.committed)
        for record in progress["candidate_dispositions"][:2]:
            self.assertEqual(record["disposition"], "ambiguous_duplicate_qr")
            self.assertEqual(record["conflicting_candidate_uids"], ["candidate_0", "candidate_1"])
            self.assertFalse(record["spatial_merge_authorized"])

    def test_pool_exhaustion_with_four_unambiguous_identities_is_terminal_incomplete(self):
        config = self.config()
        with self.assertRaises(CandidateQrGoalIncompleteError) as caught:
            execute_candidate_approach_phase(config, self.effects(
                qr_by_uid={"candidate_0": "QR_DUPLICATE", "candidate_1": "QR_DUPLICATE"},
            ))
        fields = caught.exception.to_failure_fields()
        self.assertEqual(fields["confirmed_stand_count"], 4)
        self.assertEqual(fields["expected_stand_count"], 5)
        self.assertFalse(fields["goal_completed"])
        self.assertEqual(len(self.visited), 6)
        self.assertFalse((config.session_root / "stand_facing_catalog.json").exists())
        self.assertFalse((config.session_root / "station_identity_registry.json").exists())
        pointer = json.loads((config.session_root / "candidate_goal_progress.json").read_text())
        self.assertEqual(pointer["confirmed_stand_count"], 4)

    def test_unreachable_candidate_does_not_prevent_remaining_five(self):
        config = self.config()
        result = execute_candidate_approach_phase(config, self.effects(unreachable={"candidate_0"}))
        progress, _, _ = self.assert_bound_artifacts(config, result)
        self.assertEqual(len(self.visited), 5)
        self.assertEqual(progress["candidate_dispositions"][0]["disposition"], "no_feasible_route")
        self.assertEqual(progress["unvisited_candidate_uids"], ["candidate_0"])

    def test_terminal_tf_certificate_and_motion_failures_never_skip_to_next(self):
        for failure in ("tf", "certificate", "motion"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                config = replace(self.config(), session_root=Path(tmp) / "session")
                self.visited.clear()
                self.planned_snapshots.clear()
                with self.assertRaisesRegex((RuntimeError, ValueError), "terminal"):
                    execute_candidate_approach_phase(config, self.effects(failure=failure))
                self.assertLessEqual(len(self.visited), 1)
                self.assertEqual(len(self.planned_snapshots), 1)
                self.assertFalse((config.session_root / "station_identity_registry.json").exists())

    def test_final_progress_rejects_rehashed_missing_keepout_and_duplicate_claim(self):
        config = self.config()
        result = execute_candidate_approach_phase(config, self.effects())
        _, confirmed, identity = self.assert_bound_artifacts(config, result)
        original = load_content_hashed_json(result.candidate_goal_progress_path,
                                           hash_field=GOAL_PROGRESS_HASH_FIELD)
        for field, value in (("keepout_candidate_uids", list(confirmed.candidate_uids)),
                             ("expected_stand_count", 6),
                             ("confirmed_qr_ids", ["QR_DUPLICATE"] * 5)):
            with self.subTest(field=field):
                payload = {key: item for key, item in original.items() if key != GOAL_PROGRESS_HASH_FIELD}
                payload[field] = value
                path = self.root / f"tampered_{field}.json"
                write_content_hashed_json(path, payload, hash_field=GOAL_PROGRESS_HASH_FIELD)
                with self.assertRaises(ValueError):
                    validate_candidate_qr_goal_completion(
                        path, candidate_snapshot=config.snapshot, confirmed_candidate_snapshot=confirmed,
                        observed_qr_by_candidate=identity, expected_stand_count=5,
                    )

    def test_goal_is_configured_independently_and_pool_bound_is_enforced(self):
        for goal, pool in ((0, 6), (True, 6), (5, 11), (5, 4)):
            with self.subTest(goal=goal, pool=pool), self.assertRaises(ValueError):
                CandidateQrGoalProgress((f"candidate_{i}" for i in range(pool)),
                                        expected_stand_count=goal, candidate_snapshot_sha256="a" * 64)
        original = self.config()
        config = replace(original, expected_stand_count=None,
                         plan=replace(original.plan, config=replace(original.plan.config, expected_stand_count=5)))
        result = execute_candidate_approach_phase(config, self.effects())
        self.assertEqual(result.expected_stand_count, config.plan.config.expected_stand_count)
        self.assertEqual(result.stand_count, 5)

    def test_goal_override_mismatch_blocks_all_live_effects_and_progress_publication(self):
        config = replace(self.config(), expected_stand_count=3)
        with self.assertRaisesRegex(ValueError, "sealed coverage plan"):
            execute_candidate_approach_phase(config, self.effects())
        self.assertFalse(self.visited)
        self.assertFalse(self.planned_snapshots)
        self.assertFalse(config.session_root.exists())

    def test_infeasible_preview_subset_does_not_discard_earlier_bounded_route_retry(self):
        config = self.config()
        effects = self.effects()
        original_select = effects.select_initial_preapproach
        attempts = []

        def select(request):
            if request.unresolved == frozenset({"candidate_5"}):
                raise NoFeasibleCameraCandidateError(())
            return original_select(request)

        def motion(request):
            attempts.append(request.target_id)
            if request.target_id == "candidate_4" and attempts.count("candidate_4") == 1:
                return self.fixture._route_uncertainty_rejection(request)
            return self.fixture._completed(request)

        result = execute_candidate_approach_phase(config, replace(
            effects, select_initial_preapproach=select, run_motion_leg=motion,
        ))
        progress, _, _ = self.assert_bound_artifacts(config, result)
        self.assertEqual(attempts, [f"candidate_{i}" for i in range(5)] + ["candidate_4"])
        self.assertEqual(self.visited, [f"candidate_{i}" for i in range(5)])
        self.assertEqual(progress["candidate_dispositions"][-1]["disposition"], "no_feasible_route")

    def test_unresolved_perception_advisory_survives_exhaustion_and_final_goal(self):
        config = self.config()
        suspect = config.snapshot.candidates[0]
        advisory = CandidatePerceptionAdvisory(
            kind=VISIBILITY_GAP, candidate_uid=suspect.candidate_uid,
            survey_id=config.plan.survey_id, map_bundle_sha256=config.snapshot.map_bundle_sha256,
            plan_sha256="d" * 64, viewpoint_id="survey_vp_002",
            source_morphology_sha256="e" * 64,
            candidate_frame=CandidateFrameProvenance.from_frozen_map_observation(
                map_frame="map", odom_frame="odom",
                frozen_map_point=CandidatePoint2D(suspect.geometry.x_m, suspect.geometry.y_m),
                frozen_map_from_odom=PlanarTransform2D(0.0, 0.0, 0.0),
                source_evidence_id="f" * 64,
            ),
            candidate_source_viewpoint_ids=("survey_vp_001",),
            source_observation_ids=suspect.source.observation_ids,
            proposal_max_range_m=3.5, visibility_radius_m=1.35,
            eligible_other_viewpoint_ids=(),
        )
        suspect = replace(suspect, source=replace(suspect.source, perception_advisories=(advisory,)))
        config = replace(config, snapshot=replace(config.snapshot,
                                                  candidates=(suspect, *config.snapshot.candidates[1:])))
        result = execute_candidate_approach_phase(config, self.effects(exhausted={suspect.candidate_uid}))
        progress, confirmed, identity = self.assert_bound_artifacts(config, result)
        disposition = progress["candidate_dispositions"][0]
        self.assertEqual(disposition["disposition"], "inspection_exhausted")
        self.assertEqual(disposition["perception_advisories"], [advisory.to_dict()])
        payload = dict(progress)
        payload["candidate_dispositions"][0]["perception_advisories"] = []
        bad_path = self.root / "dropped_advisory.json"
        write_content_hashed_json(bad_path, payload, hash_field=GOAL_PROGRESS_HASH_FIELD)
        with self.assertRaisesRegex(ValueError, "perception advisories"):
            validate_candidate_qr_goal_completion(
                bad_path, candidate_snapshot=config.snapshot,
                confirmed_candidate_snapshot=confirmed, observed_qr_by_candidate=identity, expected_stand_count=5,
            )
