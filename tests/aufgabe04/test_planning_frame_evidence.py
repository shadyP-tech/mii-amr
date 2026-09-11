"""Planning-frame interchange must preserve and validate admitted TF evidence."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_projection import (
    CandidatePlanningFrame,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    PlanarTransform2D,
)
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosObservation,
    RosPreflightResult,
)
from scripts.aufgabe04.real_robot.readiness.candidate_planning_frame import (
    build_candidate_planning_frame,
)


FIXTURE = Path(__file__).parent / "fixtures" / (
    "candidate_planning_frame_20260911T133125Z.json"
)
DIAGNOSTIC_FIELDS = (
    "diagnostic_chained_map_pose",
    "diagnostic_chained_map_capture",
    "chained_to_authoritative_translation_delta_m",
    "chained_to_authoritative_yaw_delta_rad",
)


def recorded_fixture():
    return json.loads(FIXTURE.read_text())


def recorded_evidence():
    return recorded_fixture()["expected_planning_frame"]


def recorded_producer(*, include_chained_capture=True):
    payload = recorded_fixture()["preflight"]
    observations = [
        RosObservation(**value) for value in payload["observations"]
        if include_chained_capture or value["name"] != "tf map->base_footprint"
    ]
    preflight = RosPreflightResult(
        **{key: value for key, value in payload.items() if key != "observations"},
        observations=observations,
    )
    return build_candidate_planning_frame(
        preflight,
        current_pose=Pose2D(**{
            key: payload["route_pose"][key]
            for key in ("x_m", "y_m", "yaw_rad")
        }),
        map_frame="map",
        odom_frame="odom",
    )


def replace_at(value, path, replacement):
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


class PlanningFrameEvidenceTests(unittest.TestCase):
    def assert_rejected(self, value):
        with self.assertRaises(ValueError):
            CandidatePlanningFrame.from_evidence(value)

    def test_legacy_frame_without_provenance_round_trips(self):
        frame = CandidatePlanningFrame(
            current_pose=Pose2D(0.25, -0.5, 0.3),
            map_from_odom=PlanarTransform2D(-1.0, 0.1, -0.2),
        )
        evidence = frame.to_evidence()
        self.assertNotIn("pose_provenance", evidence)
        restored = CandidatePlanningFrame.from_evidence(evidence)
        self.assertEqual(restored, frame)
        self.assertEqual(restored.to_evidence(), evidence)

    def test_recorded_writer_and_reader_preserve_complete_provenance(self):
        produced = recorded_producer()
        expected = recorded_evidence()
        self.assertEqual(produced.to_evidence(), expected)
        restored = CandidatePlanningFrame.from_evidence(produced.to_evidence())
        self.assertEqual(restored, produced)
        self.assertEqual(restored.to_evidence(), expected)
        # AMCL intentionally future-dates map/odom TF within its admitted window.
        self.assertLess(
            restored.pose_provenance["map_from_odom_capture"]["age_sec"], 0.0,
        )

    def test_parsing_and_serializing_detach_nested_capture_evidence(self):
        evidence = recorded_evidence()
        original = deepcopy(evidence)
        restored = CandidatePlanningFrame.from_evidence(evidence)
        evidence["pose_provenance"]["map_from_odom_capture"]["quaternion"]["w"] = 9.0
        self.assertEqual(restored.to_evidence(), original)
        exported = restored.to_evidence()
        exported["pose_provenance"]["odom_pose_capture"]["quaternion"]["w"] = 8.0
        self.assertEqual(restored.to_evidence(), original)

    def test_core_provenance_does_not_require_optional_chained_diagnostics(self):
        evidence = recorded_evidence()
        for field in DIAGNOSTIC_FIELDS:
            evidence["pose_provenance"].pop(field)
        self.assertEqual(
            CandidatePlanningFrame.from_evidence(evidence).to_evidence(), evidence,
        )

    def test_capture_metadata_remains_extensible_and_detached(self):
        evidence = recorded_evidence()
        evidence["pose_provenance"]["odom_pose_capture"]["receipt_trace"] = {
            "samples": [0.01, 0.02], "source": "saved_tf", "enabled": True,
        }
        restored = CandidatePlanningFrame.from_evidence(evidence)
        self.assertEqual(restored.to_evidence(), evidence)
        evidence["pose_provenance"]["odom_pose_capture"]["receipt_trace"]["samples"].append(0.03)
        self.assertEqual(
            restored.pose_provenance["odom_pose_capture"]["receipt_trace"]["samples"],
            [0.01, 0.02],
        )

    def test_producer_without_chained_lookup_preserves_null_diagnostic_capture(self):
        produced = recorded_producer(include_chained_capture=False)
        evidence = produced.to_evidence()
        self.assertIsNone(evidence["pose_provenance"]["diagnostic_chained_map_capture"])
        self.assertEqual(
            CandidatePlanningFrame.from_evidence(evidence).to_evidence(), evidence,
        )

    def test_reader_does_not_readmit_old_timestamps_or_gate_diagnostic_drift(self):
        # Persisted evidence is checked for internal consistency, not against
        # wall-clock time or a new runtime freshness/drift threshold.
        evidence = recorded_evidence()
        provenance = evidence["pose_provenance"]
        diagnostic = provenance["diagnostic_chained_map_pose"]
        diagnostic["x_m"] += 10.0
        provenance["diagnostic_chained_map_capture"]["x_m"] = diagnostic["x_m"]
        provenance["chained_to_authoritative_translation_delta_m"] = math.hypot(
            diagnostic["x_m"] - evidence["current_pose"]["x_m"],
            diagnostic["y_m"] - evidence["current_pose"]["y_m"],
        )
        self.assertEqual(
            CandidatePlanningFrame.from_evidence(evidence).to_evidence(), evidence,
        )

    def test_explicit_null_or_malformed_provenance_is_not_legacy_evidence(self):
        for value in (None, [], "legacy", False, 0):
            with self.subTest(value=value):
                evidence = recorded_evidence()
                evidence["pose_provenance"] = value
                self.assert_rejected(evidence)

    def test_missing_and_unknown_outer_fields_are_rejected(self):
        for field in ("current_pose", "map_from_odom", "map_frame", "odom_frame"):
            with self.subTest(field=field):
                evidence = recorded_evidence()
                evidence.pop(field)
                self.assert_rejected(evidence)
        evidence = recorded_evidence()
        evidence["motion_authorized"] = True
        self.assert_rejected(evidence)
        for value in (None, [], "frame", False):
            with self.subTest(value=value):
                self.assert_rejected(value)

    def test_missing_unknown_and_unsupported_provenance_are_rejected(self):
        for field in ("pose_basis", "map_from_odom_capture", "odom_pose_capture"):
            with self.subTest(field=field):
                evidence = recorded_evidence()
                evidence["pose_provenance"].pop(field)
                self.assert_rejected(evidence)
        for field, value in (("source", "amcl"), ("pose_basis", "chained_map_pose")):
            with self.subTest(field=field):
                evidence = recorded_evidence()
                evidence["pose_provenance"][field] = value
                self.assert_rejected(evidence)

    def test_diagnostics_must_be_a_complete_optional_group(self):
        for field in DIAGNOSTIC_FIELDS:
            with self.subTest(field=field):
                evidence = recorded_evidence()
                evidence["pose_provenance"].pop(field)
                self.assert_rejected(evidence)

    def test_nonfinite_boolean_and_malformed_pose_values_are_rejected(self):
        for owner in ("current_pose", "map_from_odom"):
            for field in ("x_m", "y_m", "yaw_rad"):
                for value in (math.nan, math.inf, -math.inf, True, False, "0.0", None):
                    with self.subTest(owner=owner, field=field, value=value):
                        evidence = recorded_evidence()
                        evidence[owner][field] = value
                        self.assert_rejected(evidence)
            for value in ({"x_m": 0.0}, {**recorded_evidence()[owner], "z_m": 0.0}):
                with self.subTest(owner=owner, value=value):
                    evidence = recorded_evidence()
                    evidence[owner] = value
                    self.assert_rejected(evidence)

    def test_capture_must_be_available_and_have_complete_finite_pose(self):
        for owner in ("map_from_odom_capture", "odom_pose_capture", "diagnostic_chained_map_capture"):
            for field, value in (
                ("available", False), ("available", 1), ("x_m", math.nan),
                ("y_m", True), ("yaw_rad", math.inf),
            ):
                with self.subTest(owner=owner, field=field):
                    evidence = recorded_evidence()
                    evidence["pose_provenance"][owner][field] = value
                    self.assert_rejected(evidence)
            evidence = recorded_evidence()
            evidence["pose_provenance"][owner].pop("x_m")
            self.assert_rejected(evidence)

    def test_capture_pose_and_stored_planning_frame_cannot_disagree(self):
        paths = (
            ("current_pose", "x_m"), ("current_pose", "yaw_rad"),
            ("map_from_odom", "y_m"), ("map_from_odom", "yaw_rad"),
            ("pose_provenance", "map_from_odom_capture", "x_m"),
            ("pose_provenance", "odom_pose_capture", "x_m"),
        )
        for path in paths:
            with self.subTest(path=path):
                evidence = recorded_evidence()
                replace_at(evidence, path, 2.0)
                self.assert_rejected(evidence)

    def test_all_capture_frame_identities_are_bound_to_the_planning_frame(self):
        for owner in ("map_from_odom_capture", "odom_pose_capture", "diagnostic_chained_map_capture"):
            for field in ("target_frame", "source_frame", "observed_target_frame", "observed_source_frame"):
                with self.subTest(owner=owner, field=field):
                    evidence = recorded_evidence()
                    evidence["pose_provenance"][owner][field] = "another_frame"
                    self.assert_rejected(evidence)
        for field, value in (("map_frame", "odom"), ("map_frame", ""), ("odom_frame", False)):
            with self.subTest(field=field, value=value):
                evidence = recorded_evidence()
                evidence[field] = value
                self.assert_rejected(evidence)

    def test_captured_frame_aliases_use_consistent_normalization(self):
        evidence = recorded_evidence()
        for owner in ("map_from_odom_capture", "odom_pose_capture", "diagnostic_chained_map_capture"):
            for field in ("target_frame", "source_frame", "observed_target_frame", "observed_source_frame"):
                capture = evidence["pose_provenance"][owner]
                capture[field] = "/" + capture[field]
        self.assertEqual(
            CandidatePlanningFrame.from_evidence(evidence).to_evidence(), evidence,
        )

    def test_capture_timestamps_and_age_are_finite_and_consistent(self):
        for owner in ("map_from_odom_capture", "odom_pose_capture", "diagnostic_chained_map_capture"):
            for field in ("stamp_sec", "capture_time_sec", "age_sec"):
                for value in (math.nan, math.inf, True, "100"):
                    with self.subTest(owner=owner, field=field, value=value):
                        evidence = recorded_evidence()
                        evidence["pose_provenance"][owner][field] = value
                        self.assert_rejected(evidence)
            for field, value in (("stamp_sec", -1.0), ("capture_time_sec", -1.0), ("age_sec", 20.0)):
                with self.subTest(owner=owner, field=field):
                    evidence = recorded_evidence()
                    evidence["pose_provenance"][owner][field] = value
                    self.assert_rejected(evidence)

    def test_diagnostic_pose_capture_and_reported_deltas_must_agree(self):
        for field, value in (
            ("chained_to_authoritative_translation_delta_m", 0.0),
            ("chained_to_authoritative_yaw_delta_rad", 0.0),
            ("chained_to_authoritative_translation_delta_m", math.nan),
            ("chained_to_authoritative_yaw_delta_rad", True),
            ("diagnostic_chained_map_pose", None),
        ):
            with self.subTest(field=field, value=value):
                evidence = recorded_evidence()
                evidence["pose_provenance"][field] = value
                self.assert_rejected(evidence)
        evidence = recorded_evidence()
        evidence["pose_provenance"]["diagnostic_chained_map_capture"]["x_m"] += 0.1
        self.assert_rejected(evidence)

    def test_optional_quaternion_and_future_tolerance_reject_invalid_numbers(self):
        for owner in ("map_from_odom_capture", "odom_pose_capture", "diagnostic_chained_map_capture"):
            for field, value in (
                ("norm", math.nan), ("norm", True), ("w", math.inf), ("z", False),
            ):
                with self.subTest(owner=owner, field=field, value=value):
                    evidence = recorded_evidence()
                    evidence["pose_provenance"][owner]["quaternion"][field] = value
                    self.assert_rejected(evidence)
            for value in (math.nan, math.inf, True, -0.1):
                with self.subTest(owner=owner, max_future_sec=value):
                    evidence = recorded_evidence()
                    evidence["pose_provenance"][owner]["max_future_sec"] = value
                    self.assert_rejected(evidence)

    def test_optional_norm_only_quaternion_evidence_is_preserved(self):
        evidence = recorded_evidence()
        for owner in ("map_from_odom_capture", "odom_pose_capture", "diagnostic_chained_map_capture"):
            evidence["pose_provenance"][owner]["quaternion"] = {"norm": 1.0}
        self.assertEqual(
            CandidatePlanningFrame.from_evidence(evidence).to_evidence(), evidence,
        )

    def test_extensible_capture_metadata_cannot_hide_nonfinite_values(self):
        evidence = recorded_evidence()
        evidence["pose_provenance"]["odom_pose_capture"]["receipt_trace"] = {
            "samples": [0.01, {"value": math.nan}],
        }
        self.assert_rejected(evidence)


if __name__ == "__main__":
    unittest.main()
