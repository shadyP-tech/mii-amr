import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.stations.arrival_pose_catalog import (  # noqa: E402
    ArrivalPoseCatalogError,
    arrival_pose_catalog_payload,
    arrival_pose_catalog_sha256,
    freeze_arrival_pose_catalog,
    load_arrival_pose_catalog,
    new_arrival_pose_catalog,
    set_expected_candidate_uids,
    upsert_arrival_pose,
    upsert_candidate_rejection,
    write_arrival_pose_catalog,
)
from scripts.aufgabe04.stations.arrival_pose_models import (  # noqa: E402
    ArrivalPoseRecord,
    ArrivalPoseValidation,
    AxisEstimate,
    CandidateRejection,
    CatalogPose2D,
    CatalogProvenance,
    FaceSelection,
    StandEstimate,
)


MAP_HASH = "a" * 64
WORLD_HASH = "b" * 64


def _provenance(**changes):
    values = {
        "planning_frame": "odom",
        "map_yaml_sha256": MAP_HASH,
        "world_id": "aufgabe04_stands.world",
        "world_sha256": WORLD_HASH,
        "session_id": "gazebo-001",
        "environment": "simulation",
    }
    values.update(changes)
    return CatalogProvenance(**values)


def _record(candidate_uid="candidate-a", observation_id="obs-001", x_m=0.0):
    # Axis points along +x, therefore the selected face normal points along +y.
    return ArrivalPoseRecord(
        candidate_uid=candidate_uid,
        stand_id="stand-a" if candidate_uid == "candidate-a" else "stand-b",
        stand=StandEstimate(x_m=x_m, y_m=0.0, radius_m=0.06, uncertainty_m=0.02),
        axis=AxisEstimate(
            axis_rad=0.0,
            confidence=0.92,
            sample_count=8,
            estimator="silhouette/head_rectangle",
            observation_unix_sec=101.0,
        ),
        face=FaceSelection(
            face_id="face-0",
            outward_normal_rad=math.pi / 2.0,
            identity_resolved=False,
            evidence_kind="robot_facing_side",
            evidence_confidence=0.83,
            evidence_hard=False,
            evidence_valid=True,
            evidence_provenance="synchronized/lidar_camera",
        ),
        arrival_pose=CatalogPose2D(x_m=x_m, y_m=0.32, yaw_rad=-math.pi / 2.0),
        corridor_entry_pose=CatalogPose2D(
            x_m=x_m, y_m=0.72, yaw_rad=-math.pi / 2.0
        ),
        standoff_m=0.32,
        corridor_length_m=0.40,
        validation=ArrivalPoseValidation(
            target_in_bounds=True,
            target_collision_free=True,
            corridor_collision_free=True,
            validated_map_yaml_sha256=MAP_HASH,
            validated_unix_sec=102.0,
        ),
        source_observation_ids=(observation_id,),
        sensor_stamp_sec=77.0,
        source="simulation/synchronized_viewpoint",
    )


class ArrivalPoseCatalogTest(unittest.TestCase):
    def _catalog(self, expected=("candidate-a", "candidate-b")):
        return new_arrival_pose_catalog(
            catalog_id="survey-001",
            provenance=_provenance(),
            expected_candidate_uids=expected,
            created_unix_sec=100.0,
        )

    def test_round_trip_is_strict_hashed_and_atomic(self):
        catalog = upsert_arrival_pose(
            self._catalog(("candidate-a",)), _record(), updated_unix_sec=103.0
        )
        catalog = freeze_arrival_pose_catalog(catalog, frozen_unix_sec=104.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "arrival_pose_catalog.json"
            written_hash = write_arrival_pose_catalog(path, catalog)
            loaded = load_arrival_pose_catalog(path, required_provenance=_provenance())
            payload = json.loads(path.read_text())
            temporary_files = tuple(path.parent.glob(f".{path.name}.*.tmp"))

        self.assertEqual(loaded, catalog)
        self.assertEqual(written_hash, arrival_pose_catalog_sha256(catalog))
        self.assertEqual(payload["catalog_sha256"], written_hash)
        self.assertTrue(payload["frozen"])
        self.assertTrue(path.name.endswith(".json"))
        self.assertEqual(temporary_files, ())

    def test_upsert_is_idempotent_and_conflicting_geometry_is_rejected(self):
        initial = self._catalog()
        first = upsert_arrival_pose(initial, _record(), updated_unix_sec=103.0)
        retry = upsert_arrival_pose(first, _record(), updated_unix_sec=104.0)

        self.assertIs(retry, first)
        self.assertEqual(first.revision, 1)
        with self.assertRaises(ArrivalPoseCatalogError) as raised:
            upsert_arrival_pose(
                first,
                replace(_record(), sensor_stamp_sec=78.0),
                updated_unix_sec=104.0,
            )
        self.assertEqual(raised.exception.code, "candidate_conflict")

    def test_observation_ancestry_cannot_belong_to_two_candidates(self):
        catalog = upsert_arrival_pose(
            self._catalog(), _record(), updated_unix_sec=103.0
        )
        with self.assertRaises(ArrivalPoseCatalogError) as raised:
            upsert_arrival_pose(
                catalog,
                _record(candidate_uid="candidate-b", observation_id="obs-001", x_m=1.0),
                updated_unix_sec=104.0,
            )
        self.assertEqual(raised.exception.code, "observation_identity_conflict")

    def test_catalog_freezes_only_after_every_candidate_is_resolved(self):
        catalog = upsert_arrival_pose(
            self._catalog(), _record(), updated_unix_sec=103.0
        )
        with self.assertRaises(ArrivalPoseCatalogError) as raised:
            freeze_arrival_pose_catalog(catalog, frozen_unix_sec=104.0)
        self.assertEqual(raised.exception.code, "catalog_incomplete")

        rejection = CandidateRejection(
            candidate_uid="candidate-b",
            reason="no collision-free terminal corridor",
            source_observation_ids=("obs-002",),
            rejected_unix_sec=104.0,
        )
        catalog = upsert_candidate_rejection(
            catalog, rejection, updated_unix_sec=104.0
        )
        self.assertTrue(catalog.complete)
        frozen = freeze_arrival_pose_catalog(catalog, frozen_unix_sec=105.0)

        self.assertTrue(frozen.frozen)
        self.assertEqual(frozen.revision, 3)
        self.assertEqual(frozen.frozen_unix_sec, 105.0)
        self.assertIs(freeze_arrival_pose_catalog(frozen, frozen_unix_sec=106.0), frozen)
        with self.assertRaises(ArrivalPoseCatalogError) as raised:
            upsert_arrival_pose(
                frozen,
                _record(candidate_uid="candidate-b", observation_id="obs-003", x_m=1.0),
                updated_unix_sec=106.0,
            )
        self.assertEqual(raised.exception.code, "candidate_conflict")

    def test_open_catalog_can_set_expected_candidates_once_observation_closes(self):
        catalog = self._catalog(())
        catalog = upsert_arrival_pose(catalog, _record(), updated_unix_sec=103.0)
        self.assertFalse(catalog.complete)
        sealed = set_expected_candidate_uids(
            catalog, ("candidate-a",), updated_unix_sec=104.0
        )
        self.assertTrue(sealed.complete)
        self.assertEqual(sealed.revision, 2)

    def test_loader_rejects_hash_tampering_unknown_fields_and_provenance(self):
        catalog = upsert_arrival_pose(
            self._catalog(("candidate-a",)), _record(), updated_unix_sec=103.0
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.json"
            write_arrival_pose_catalog(path, catalog)
            payload = json.loads(path.read_text())
            payload["revision"] = 99
            path.write_text(json.dumps(payload))
            with self.assertRaises(ArrivalPoseCatalogError) as tampered:
                load_arrival_pose_catalog(path)
            self.assertEqual(tampered.exception.code, "hash_mismatch")

            payload = arrival_pose_catalog_payload(catalog)
            payload["unknown"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaises(ArrivalPoseCatalogError) as unknown:
                load_arrival_pose_catalog(path)
            self.assertEqual(unknown.exception.code, "catalog_corrupt")

            write_arrival_pose_catalog(path, catalog)
            with self.assertRaises(ArrivalPoseCatalogError) as mismatch:
                load_arrival_pose_catalog(
                    path,
                    required_provenance=_provenance(session_id="gazebo-002"),
                )
            self.assertEqual(mismatch.exception.code, "provenance_mismatch")

    def test_geometry_and_map_validation_fail_closed(self):
        catalog = self._catalog(("candidate-a",))
        bad_yaw = replace(
            _record(),
            arrival_pose=CatalogPose2D(0.0, 0.32, math.pi / 2.0),
        )
        with self.assertRaises(ArrivalPoseCatalogError) as geometry:
            upsert_arrival_pose(catalog, bad_yaw, updated_unix_sec=103.0)
        self.assertEqual(geometry.exception.code, "geometry_mismatch")

        bad_map = replace(
            _record(),
            validation=replace(
                _record().validation, validated_map_yaml_sha256="c" * 64
            ),
        )
        with self.assertRaises(ArrivalPoseCatalogError) as provenance:
            upsert_arrival_pose(catalog, bad_map, updated_unix_sec=103.0)
        self.assertEqual(provenance.exception.code, "provenance_mismatch")

    def test_in_memory_construction_does_not_coerce_identifiers_or_booleans(self):
        with self.assertRaises(ArrivalPoseCatalogError):
            new_arrival_pose_catalog(
                catalog_id="survey-001",
                provenance=_provenance(),
                expected_candidate_uids=(1,),  # type: ignore[arg-type]
                created_unix_sec=100.0,
            )
        with self.assertRaises(ArrivalPoseCatalogError):
            new_arrival_pose_catalog(
                catalog_id="survey-001",
                provenance=_provenance(),
                expected_candidate_uids=("candidate-a",),
                created_unix_sec=True,
            )


if __name__ == "__main__":
    unittest.main()
