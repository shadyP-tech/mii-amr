from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.aufgabe04.artifacts.candidate_perception_advisory import (
    MORPHOLOGY_CONFLICT, VISIBILITY_GAP, advisory_from_payload, advisory_payload,
    validate_candidate_perception_advisory,
)
from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.foundation.content_hashed_evidence import write_content_hashed_json
from scripts.aufgabe04.navigation.coverage.coverage_stop_perception_admission import (
    CoverageEpochPerceptionAdmission, _content_hashed_artifact,
    coverage_stop_perception_summary_fields, prepare_coverage_morphology_conflicts,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_static_map_admission import (
    evaluate_stand_candidate_static_map_admission,
)
from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance, CandidatePoint2D,
)
from scripts.aufgabe04.navigation.coverage.coverage_morphology_conflict import (
    retain_candidate_perception_advisories,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STAND_SURVEY_REGISTRY_SCHEMA_VERSION, StandSurveyRegistry, SurveyCandidate,
    load_stand_survey_registry, stand_survey_registry_payload,
    stand_survey_registry_sha256, write_stand_survey_registry,
)
from scripts.aufgabe04.navigation.planning.costmap import Costmap
from scripts.aufgabe04.perception.lidar_stand_morphology import (
    StandMorphologyAdmission, StandMorphologyEvidence, assess_stand_width_samples,
    stand_width_profile_from_radius,
)
from scripts.aufgabe04.perception.stand_confirmation import ConfirmedStand
from tests.aufgabe04.test_coverage_candidate_reconciliation import _grid, _plan


def _recorded_case():
    fixture = json.loads((Path(__file__).parent / "fixtures/lidar_morphology_conflict_20260907.json").read_text())
    raw = fixture["candidate"]
    candidate = SurveyCandidate(**{
        **raw, "source_observation_ids": tuple(raw["source_observation_ids"]),
        "viewpoint_ids": tuple(raw["viewpoint_ids"]),
        "frame_provenance": CandidateFrameProvenance.from_mapping(raw["frame_provenance"]),
    })
    raw_track = fixture["rejected_track"]
    track = ConfirmedStand(
        stand_id=raw_track["stand_id"], x_m=raw_track["x_m"], y_m=raw_track["y_m"],
        confidence=0.8, hit_count=len(raw_track["widths_m"]),
        first_seen_sec=3.0, last_seen_sec=4.0, first_confirmed_at_sec=3.1,
        source_observation_ids=tuple(raw_track["source_observation_ids"]),
        provenance=raw_track["provenance"],
    )
    profile = stand_width_profile_from_radius(candidate.radius_m)
    assessment = assess_stand_width_samples(raw_track["widths_m"], profile=profile)
    morphology = StandMorphologyAdmission(
        schema_version=1, profile=profile, source_observation_count=track.hit_count,
        evidence=(StandMorphologyEvidence(track.stand_id, track.source_observation_ids, assessment),),
        admitted_stands=(), rejected_stands=(track,),
    )
    plan = _plan(check_visible=False)
    plan = replace(
        plan, survey_id=fixture["source_run"], map_bundle_sha256=fixture["map_bundle_sha256"],
        viewpoints=tuple(replace(v, viewpoint_id=f"survey_vp_{i:03}") for i, v in enumerate(plan.viewpoints, 1)),
    )
    registry = StandSurveyRegistry(
        schema_version=STAND_SURVEY_REGISTRY_SCHEMA_VERSION, survey_id=plan.survey_id,
        planning_frame=plan.planning_frame, map_bundle_sha256=plan.map_bundle_sha256,
        candidates=(candidate,),
    )
    return registry, plan, morphology, fixture


def _evaluate(*, registry=None, plan=None, morphology=None, prior_registry=None):
    default_registry, default_plan, default_morphology, fixture = _recorded_case()
    return retain_candidate_perception_advisories(
        registry or default_registry, plan=plan or default_plan,
        static_costmap=Costmap.from_occupancy_grid(_grid()),
        morphology=morphology or default_morphology, viewpoint_id="survey_vp_002",
        morphology_sha256=fixture["source_morphology_sha256"], proposal_max_range_m=3.5,
        prior_registry=prior_registry,
    )


class CoverageMorphologyConflictTests(unittest.TestCase):
    def test_recorded_rejected_neighbor_survives_as_unresolved_evidence(self):
        registry, _, morphology, _ = _recorded_case()
        result = _evaluate()
        candidate = result.updated_registry.candidates[0]
        conflict = next(a for a in candidate.perception_advisories if a.kind == MORPHOLOGY_CONFLICT)
        self.assertAlmostEqual(conflict.association_distance_m, 0.1071, places=4)
        self.assertEqual(conflict.track_id, "detected_stand_02")
        self.assertEqual(len(conflict.track_source_observation_ids), 81)
        self.assertIn("median_width_above_maximum", conflict.rejection_reasons)
        self.assertEqual(conflict.possible_candidate_uids, (candidate.candidate_uid,))
        self.assertEqual(replace(candidate, perception_advisories=()), registry.candidates[0])
        self.assertEqual(morphology.admitted_stands, ())
        self.assertFalse(result.payload["candidate_rejection_authorized"])
        self.assertFalse(result.payload["keepouts_changed"])
        self.assertFalse(result.payload["motion_authorized"])

    def test_visibility_range_gap_is_explicit_without_claiming_absence(self):
        gap = next(a for a in _evaluate().advisories if a.kind == VISIBILITY_GAP)
        self.assertEqual(gap.eligible_other_viewpoint_ids, ())
        self.assertEqual((gap.proposal_max_range_m, gap.visibility_radius_m), (3.5, 1.35))
        self.assertEqual(gap.to_dict()["status"], "unresolved")
        self.assertIsNone(gap.track_id)

    def test_ambiguous_neighbors_retain_both_keepouts_without_identity_claim(self):
        registry, plan, _, _ = _recorded_case()
        original = registry.candidates[0]
        frame = original.frame_provenance
        shifted = CandidateFrameProvenance.from_frozen_map_observation(
            map_frame=frame.map_frame, odom_frame=frame.odom_frame,
            frozen_map_point=CandidatePoint2D(original.x_m + 0.02, original.y_m),
            frozen_map_from_odom=frame.frozen_map_from_odom, source_evidence_id="f" * 64,
        )
        other = replace(original, candidate_uid="survey_candidate_0005", x_m=original.x_m + 0.02,
                        frame_provenance=shifted, source_observation_ids=("other_source",))
        registry = replace(registry, candidates=(original, other))
        result = _evaluate(registry=registry, plan=plan)
        conflicts = [a for a in result.advisories if a.kind == MORPHOLOGY_CONFLICT]
        self.assertEqual(len(conflicts), 2)
        self.assertTrue(all(len(a.possible_candidate_uids) == 2 for a in conflicts))
        self.assertEqual(result.payload["rejected_track_dispositions"][0]["association_status"], "ambiguous_spatial_neighbors")
        self.assertEqual(tuple(replace(c, perception_advisories=()) for c in result.updated_registry.candidates), registry.candidates)

    def test_far_track_does_not_expand_existing_association_radius(self):
        _, _, morphology, _ = _recorded_case()
        track = morphology.rejected_stands[0]
        result = _evaluate(morphology=replace(morphology, rejected_stands=(replace(track, x_m=track.x_m + 0.5),)))
        self.assertFalse(any(a.kind == MORPHOLOGY_CONFLICT for a in result.advisories))
        self.assertEqual(result.payload["rejected_track_dispositions"][0]["association_status"], "unassociated")

    def test_current_positive_fusion_does_not_hide_historical_morphology_conflict(self):
        before, plan, _, _ = _recorded_case()
        candidate = before.candidates[0]
        after = replace(before, candidates=(replace(
            candidate, viewpoint_ids=(*candidate.viewpoint_ids, "survey_vp_002"),
            source_observation_ids=(*candidate.source_observation_ids, "current_accepted_track"),
            status="pending_camera",
        ),))
        result = _evaluate(registry=after, prior_registry=before, plan=plan)
        conflict = next(a for a in result.advisories if a.kind == MORPHOLOGY_CONFLICT)
        self.assertEqual(conflict.candidate_source_viewpoint_ids, ("survey_vp_001",))
        self.assertEqual(conflict.candidate_frame, candidate.frame_provenance)
        self.assertEqual(result.updated_registry.candidates[0].status, "pending_camera")

    def test_malformed_declared_frozen_provenance_fails_closed(self):
        _, _, morphology, _ = _recorded_case()
        track = morphology.rejected_stands[0]
        provenance = json.loads(json.dumps(track.provenance))
        del provenance["provenance"]["runtime_config"]["frozen_odom_observation_geometry"]["map_from_odom"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            _evaluate(morphology=replace(morphology, rejected_stands=(replace(track, provenance=provenance),)))

    def test_mixed_missing_frame_is_not_a_map_distance_fallback(self):
        registry, plan, _, _ = _recorded_case()
        registry = replace(registry, candidates=(replace(registry.candidates[0], frame_provenance=None),))
        with self.assertRaisesRegex(ValueError, "mixed legacy"):
            _evaluate(registry=registry, plan=plan)

    def test_advisory_round_trip_hash_uid_and_geometry_are_validated(self):
        advisory = next(a for a in _evaluate().advisories if a.kind == MORPHOLOGY_CONFLICT)
        payload = advisory_payload(advisory)
        self.assertEqual(advisory_from_payload(payload), advisory)
        with self.assertRaisesRegex(ValueError, "another candidate"):
            validate_candidate_perception_advisory(advisory, candidate_uid="different")
        changed = {**payload, "association_distance_m": 0.01}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            advisory_from_payload(changed)
        changed.pop("advisory_sha256")
        changed["advisory_sha256"] = payload_sha256(changed)
        with self.assertRaisesRegex(ValueError, "distance differs"):
            advisory_from_payload(changed)

    def test_registry_round_trip_preserves_advisories_and_legacy_empty_hash(self):
        registry, plan, _, _ = _recorded_case()
        original_payload = stand_survey_registry_payload(registry)
        self.assertNotIn("perception_advisories", original_payload["candidates"][0])
        self.assertEqual(stand_survey_registry_sha256(registry), payload_sha256(original_payload))
        updated = _evaluate().updated_registry
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.json"
            write_stand_survey_registry(path, updated, plan)
            self.assertEqual(load_stand_survey_registry(path, plan), updated)
        self.assertNotEqual(stand_survey_registry_sha256(registry), stand_survey_registry_sha256(updated))

    def test_replaying_same_advisory_does_not_duplicate_historical_evidence(self):
        first = _evaluate()
        second = _evaluate(registry=first.updated_registry)
        self.assertEqual(second.updated_registry, first.updated_registry)

    def test_stopped_epoch_binds_immutable_artifact_and_rejects_source_tampering(self):
        registry, plan, morphology, _ = _recorded_case()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _content_hashed_artifact(
                kind="morphology", survey_root=root, viewpoint_id="survey_vp_002",
                filename_label="morphology", hash_field="morphology_sha256",
                payload={**morphology.to_evidence_dict(), "observer_contract": {
                    "proposal_detector_config": {"max_range_m": 3.5},
                }},
            )
            admission = CoverageEpochPerceptionAdmission(
                raw_stands=morphology.rejected_stands, morphology_admission=morphology,
                static_map_admission=evaluate_stand_candidate_static_map_admission(
                    Costmap.from_occupancy_grid(_grid()), (),
                    candidate_radius_m=0.06, candidate_uncertainty_m=0.02,
                ), visibility_evidence=None, morphology_artifact=artifact,
                static_map_artifact=artifact,
            )
            def prepare():
                return prepare_coverage_morphology_conflicts(
                    survey_root=root, registry=registry, plan=plan, viewpoint_id="survey_vp_002",
                    occupancy_grid=_grid(), epoch_admission=admission,
                )
            prepared = prepare()
            output = prepared.artifact
            self.assertEqual(
                write_content_hashed_json(output.path, output.payload, hash_field=output.hash_field),
                output.sha256,
            )
            saved = json.loads(output.path.read_text())
            self.assertEqual(saved[output.hash_field], output.sha256)
            self.assertEqual(saved["source_morphology_sha256"], artifact.sha256)
            summary = coverage_stop_perception_summary_fields(admission, None, prepared)
            self.assertEqual(summary["candidate_perception_advisory_count"], 2)
            self.assertEqual(summary["candidate_perception_advisories_sha256"], output.sha256)
            artifact.payload["observer_contract"]["proposal_detector_config"]["max_range_m"] = 4.0
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                prepare()

    def test_advisory_payload_cannot_claim_rejection_or_motion_authority(self):
        payload = _evaluate().advisories[0].to_dict()
        for field in ("motion_authorized", "candidate_rejection_authorized"):
            changed = {**payload, field: True}
            changed.pop("advisory_sha256")
            changed["advisory_sha256"] = payload_sha256(changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "invalid advisory"):
                advisory_from_payload(changed)


if __name__ == "__main__":
    unittest.main()
