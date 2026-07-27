import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.qr_scanning.events import (  # noqa: E402
    QRConsensusEvidence,
    QRGeometryEvidence,
    QRObservationEvent,
    QRValidationPolicy,
    StationIdentityRegistry,
    validate_qr_observation,
)
from scripts.aufgabe04.stations.station_identity_registry import (  # noqa: E402
    StationIdentity,
    load_station_identity_registry,
    new_station_identity_registry,
    write_station_identity_registry,
)


CALIBRATION_SHA256 = "a" * 64


def _persisted_registry():
    return new_station_identity_registry(
        registry_id="qr_event_test_registry",
        created_unix_sec=1.0,
        candidate_snapshot_sha256="c" * 64,
        source_artifact_sha256="d" * 64,
        expected_candidate_uids=("candidate_a", "candidate_b"),
        mappings=(
            StationIdentity("candidate_a", "A", "station_A"),
            StationIdentity("candidate_b", "B", "station_B"),
        ),
    )


def _registry():
    return StationIdentityRegistry(_persisted_registry())


def _event(
    *,
    event_id="event-1",
    qr_id="A",
    station_id="station_A",
    candidate_uid="candidate_a",
    observed_at_sec=99.8,
    received_at_sec=99.9,
):
    return QRObservationEvent(
        event_id=event_id,
        robot_id="robot_1",
        qr_id=qr_id,
        station_id=station_id,
        candidate_uid=candidate_uid,
        observed_at_sec=observed_at_sec,
        received_at_sec=received_at_sec,
        clock_id="unix_epoch",
        source="onboard_camera",
        source_frame_id="camera_optical_frame",
        confidence=0.95,
        geometry=QRGeometryEvidence.create(
            image_width_px=640,
            image_height_px=480,
            corners_px=((100.0, 100.0), (140.0, 100.0), (140.0, 140.0), (100.0, 140.0)),
        ),
        consensus=QRConsensusEvidence.create(
            qr_id=qr_id,
            sample_ids=(f"{event_id}-s1", f"{event_id}-s2", f"{event_id}-s3"),
            agreeing_sample_ids=(f"{event_id}-s1", f"{event_id}-s2", f"{event_id}-s3"),
            window_start_sec=observed_at_sec - 0.2,
            window_end_sec=observed_at_sec,
        ),
        calibration_sha256=CALIBRATION_SHA256,
    )


class QREventsTest(unittest.TestCase):
    def test_registry_resolves_qr_A_to_canonical_station_A(self):
        registry = _registry()
        validated = validate_qr_observation(
            _event(),
            registry=registry,
            now_sec=100.0,
            policy=QRValidationPolicy(
                expected_calibration_sha256=CALIBRATION_SHA256,
                expected_clock_id="unix_epoch",
            ),
            expected_robot_id="robot_1",
        )

        self.assertEqual(validated.identity.qr_id, "A")
        self.assertEqual(validated.identity.station_id, "station_A")
        self.assertEqual(validated.identity.candidate_uid, "candidate_a")
        self.assertIs(validated.identity.persisted, registry.persisted_registry.for_qr("A"))

    def test_loaded_registry_is_the_only_mapping_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "station_identity_registry.json"
            write_station_identity_registry(path, _persisted_registry())
            loaded = load_station_identity_registry(path)
            registry = StationIdentityRegistry(loaded)

        validated = validate_qr_observation(
            _event(),
            registry=registry,
            now_sec=100.0,
        )

        self.assertIs(registry.persisted_registry, loaded)
        self.assertIs(validated.identity.persisted, loaded.for_qr("A"))
        self.assertEqual(validated.identity.server_station_id, "station_A")
        self.assertEqual(validated.identity.candidate_uid, "candidate_a")
        with self.assertRaisesRegex(ValueError, "unknown server station"):
            registry.resolve("A")

    def test_rejects_cross_station_identity_fields(self):
        with self.assertRaisesRegex(ValueError, "identity fields disagree"):
            validate_qr_observation(
                _event(candidate_uid="candidate_b"),
                registry=_registry(),
                now_sec=100.0,
            )

    def test_rejects_stale_future_and_replayed_events(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_qr_observation(
                _event(observed_at_sec=90.0, received_at_sec=90.1),
                registry=_registry(),
                now_sec=100.0,
            )
        with self.assertRaisesRegex(ValueError, "future"):
            validate_qr_observation(
                _event(observed_at_sec=101.0, received_at_sec=101.0),
                registry=_registry(),
                now_sec=100.0,
            )
        with self.assertRaisesRegex(ValueError, "replayed"):
            validate_qr_observation(
                _event(event_id="already-seen"),
                registry=_registry(),
                now_sec=100.0,
                seen_event_ids=("already-seen",),
            )

    def test_rejects_geometry_digest_that_does_not_bind_corners(self):
        valid = QRGeometryEvidence.create(
            image_width_px=100,
            image_height_px=100,
            corners_px=((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            QRGeometryEvidence(
                image_width_px=100,
                image_height_px=100,
                corners_px=((11.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)),
                geometry_sha256=valid.geometry_sha256,
            )

        with self.assertRaisesRegex(ValueError, "ordered convex"):
            QRGeometryEvidence.create(
                image_width_px=100,
                image_height_px=100,
                corners_px=((10.0, 10.0), (20.0, 10.0), (14.0, 14.0), (10.0, 20.0)),
            )

    def test_rejects_wrong_calibration_and_weak_consensus(self):
        with self.assertRaisesRegex(ValueError, "calibration hash"):
            validate_qr_observation(
                _event(),
                registry=_registry(),
                now_sec=100.0,
                policy=QRValidationPolicy(expected_calibration_sha256="b" * 64),
            )

        weak = _event(event_id="weak")
        with self.assertRaisesRegex(ValueError, "too few consensus"):
            validate_qr_observation(
                weak,
                registry=_registry(),
                now_sec=100.0,
                policy=QRValidationPolicy(min_consensus_samples=4),
            )

    def test_rejects_mixed_clock_domain(self):
        with self.assertRaisesRegex(ValueError, "clock_id"):
            validate_qr_observation(
                _event(),
                registry=_registry(),
                now_sec=100.0,
                policy=QRValidationPolicy(expected_clock_id="ros_sim_time"),
            )


if __name__ == "__main__":
    unittest.main()
