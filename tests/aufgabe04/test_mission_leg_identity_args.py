import unittest
from types import SimpleNamespace

from scripts.aufgabe04.navigation.mission_leg_identity_args import (
    build_mission_leg_event_fields,
    resolve_explicit_mission_leg_evidence_identity,
    resolve_mission_leg_event_identity,
    resolve_startup_reseal_permit_identity,
)
from scripts.aufgabe04.navigation.mission_leg_motion_permit import MissionLegKind


def identity_args(**overrides):
    values = {
        "mission_leg_evidence_kind": None,
        "mission_leg_evidence_index": None,
        "mission_leg_evidence_target_id": "",
        "mission_leg_kind": None,
        "mission_leg_index": None,
        "mission_leg_target_id": "",
        "startup_reseal_mission_leg_kind": None,
        "startup_reseal_mission_leg_index": None,
        "startup_reseal_target_id": "",
        "coverage_transient_replan_enabled": False,
        "coverage_transient_replan_leg_index": None,
        "coverage_transient_replan_target_viewpoint_id": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MissionLegIdentityArgsTest(unittest.TestCase):
    def test_missing_identity_retains_empty_coverage_event_aliases(self):
        self.assertIsNone(resolve_mission_leg_event_identity(identity_args()))
        self.assertEqual(
            build_mission_leg_event_fields(identity_args()),
            {
                "coverage_leg_index": None,
                "target_viewpoint_id": "",
            },
        )

    def test_candidate_evidence_emits_generic_fields_without_coverage_aliases(self):
        args = identity_args(
            mission_leg_evidence_kind="candidate_preapproach",
            mission_leg_evidence_index=4,
            mission_leg_evidence_target_id="  stand-5  ",
        )

        self.assertEqual(
            resolve_explicit_mission_leg_evidence_identity(args),
            (MissionLegKind.CANDIDATE_PREAPPROACH, 4, "stand-5"),
        )
        self.assertEqual(
            build_mission_leg_event_fields(args),
            {
                "mission_leg_kind": "candidate_preapproach",
                "mission_leg_index": 4,
                "target_id": "stand-5",
                "coverage_leg_index": None,
                "target_viewpoint_id": "",
            },
        )

    def test_coverage_identity_retains_legacy_event_aliases(self):
        args = identity_args(
            mission_leg_evidence_kind="coverage",
            mission_leg_evidence_index=2,
            mission_leg_evidence_target_id="survey-vp-3",
            coverage_transient_replan_enabled=True,
            coverage_transient_replan_leg_index=2,
            coverage_transient_replan_target_viewpoint_id="survey-vp-3",
        )

        self.assertEqual(
            build_mission_leg_event_fields(args),
            {
                "mission_leg_kind": "coverage",
                "mission_leg_index": 2,
                "target_id": "survey-vp-3",
                "coverage_leg_index": 2,
                "target_viewpoint_id": "survey-vp-3",
            },
        )

    def test_consistent_evidence_permit_and_startup_identities_coalesce(self):
        args = identity_args(
            mission_leg_evidence_kind="opposite_face",
            mission_leg_evidence_index=1,
            mission_leg_evidence_target_id="stand-2",
            mission_leg_kind="opposite_face",
            mission_leg_index=1,
            mission_leg_target_id="stand-2",
            startup_reseal_mission_leg_kind="opposite_face",
            startup_reseal_mission_leg_index=1,
            startup_reseal_target_id="stand-2",
        )

        self.assertEqual(
            resolve_mission_leg_event_identity(args),
            (MissionLegKind.OPPOSITE_FACE, 1, "stand-2"),
        )

    def test_conflicting_identity_sources_are_rejected(self):
        args = identity_args(
            mission_leg_evidence_kind="candidate_preapproach",
            mission_leg_evidence_index=1,
            mission_leg_evidence_target_id="stand-1",
            startup_reseal_mission_leg_kind="candidate_preapproach",
            startup_reseal_mission_leg_index=1,
            startup_reseal_target_id="stand-2",
        )

        with self.assertRaisesRegex(
            ValueError,
            "conflicting mission-leg evidence identities",
        ):
            resolve_mission_leg_event_identity(args)

    def test_partial_or_boolean_evidence_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            resolve_explicit_mission_leg_evidence_identity(
                identity_args(mission_leg_evidence_kind="coverage")
            )
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            resolve_explicit_mission_leg_evidence_identity(
                identity_args(
                    mission_leg_evidence_kind="coverage",
                    mission_leg_evidence_index=True,
                    mission_leg_evidence_target_id="survey-vp-1",
                )
            )

    def test_startup_permit_supports_legacy_coverage_identity(self):
        permit = SimpleNamespace(
            leg_index=3,
            target_viewpoint_id="survey-vp-4",
        )

        self.assertEqual(
            resolve_startup_reseal_permit_identity(permit),
            (MissionLegKind.COVERAGE, 3, "survey-vp-4"),
        )

    def test_startup_permit_requires_matching_generic_and_legacy_aliases(self):
        permit = SimpleNamespace(
            mission_leg_kind=MissionLegKind.CANDIDATE_PREAPPROACH,
            mission_leg_index=2,
            target_id="stand-3",
            leg_index=2,
            target_viewpoint_id="stand-3",
        )
        self.assertEqual(
            resolve_startup_reseal_permit_identity(permit),
            (MissionLegKind.CANDIDATE_PREAPPROACH, 2, "stand-3"),
        )

        permit.target_viewpoint_id = "stand-4"
        with self.assertRaisesRegex(ValueError, "aliases mismatch"):
            resolve_startup_reseal_permit_identity(permit)

        del permit.target_id
        with self.assertRaisesRegex(ValueError, "partial mission-leg identity"):
            resolve_startup_reseal_permit_identity(permit)


if __name__ == "__main__":
    unittest.main()
