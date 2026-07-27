import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.logistics.mission_controller import (  # noqa: E402
    MissionControlError,
    MissionController,
    MissionControllerPolicy,
)
from scripts.aufgabe04.logistics.models import MissionState  # noqa: E402
from scripts.aufgabe04.logistics.server_validation.models import (  # noqa: E402
    ValidatedServerTask,
    server_order_sha256,
)
from scripts.aufgabe04.qr_scanning.events import (  # noqa: E402
    QRConsensusEvidence,
    QRGeometryEvidence,
    QRObservationEvent,
    QRValidationPolicy,
    StationIdentityRegistry,
)
from scripts.aufgabe04.stations.station_identity_registry import (  # noqa: E402
    StationIdentity,
    new_station_identity_registry,
)


CALIBRATION_SHA256 = "a" * 64


def _registry():
    persisted = new_station_identity_registry(
        registry_id="mission_controller_test_registry",
        created_unix_sec=1.0,
        candidate_snapshot_sha256="c" * 64,
        source_artifact_sha256="d" * 64,
        expected_candidate_uids=(
            "candidate_a",
            "candidate_b",
            "candidate_c",
            "candidate_home",
        ),
        mappings=(
            StationIdentity("candidate_home", "HOME", "station_HOME"),
            StationIdentity("candidate_a", "A", "station_A"),
            StationIdentity("candidate_b", "B", "station_B"),
            StationIdentity("candidate_c", "C", "station_C"),
        ),
    )
    return StationIdentityRegistry(persisted)


def _event(event_id, qr_id, station_id, candidate_uid, observed_at_sec):
    return QRObservationEvent(
        event_id=event_id,
        robot_id="robot_1",
        qr_id=qr_id,
        station_id=station_id,
        candidate_uid=candidate_uid,
        observed_at_sec=observed_at_sec,
        received_at_sec=observed_at_sec + 0.02,
        clock_id="unix_epoch",
        source="onboard_camera",
        source_frame_id="camera_optical_frame",
        confidence=0.99,
        geometry=QRGeometryEvidence.create(
            image_width_px=640,
            image_height_px=480,
            corners_px=((100.0, 100.0), (140.0, 100.0), (140.0, 140.0), (100.0, 140.0)),
        ),
        consensus=QRConsensusEvidence.create(
            qr_id=qr_id,
            sample_ids=(f"{event_id}-s1", f"{event_id}-s2", f"{event_id}-s3"),
            agreeing_sample_ids=(f"{event_id}-s1", f"{event_id}-s2", f"{event_id}-s3"),
            window_start_sec=observed_at_sec - 0.1,
            window_end_sec=observed_at_sec,
        ),
        calibration_sha256=CALIBRATION_SHA256,
    )


def _task(
    *,
    now_sec=100.0,
    ordered_station_ids=("station_A", "station_C", "station_B"),
    target_station="station_A",
):
    plan_generated_at_sec = now_sec - 2.0
    digest = server_order_sha256(
        robot_id="robot_1",
        mission_id="mission-1",
        target_station=target_station,
        plan_step_index=4,
        ordered_station_ids=ordered_station_ids,
        plan_generated_at_sec=plan_generated_at_sec,
    )
    return ValidatedServerTask(
        robot_id="robot_1",
        mission_id="mission-1",
        state="ACTIVE",
        last_qr="HOME",
        resolved_current_station="station_HOME",
        target_station=target_station,
        cargo="PUCK",
        plan_step_index=4,
        evidence={},
        ordered_station_ids=tuple(ordered_station_ids),
        status_observed_at_sec=now_sec - 1.0,
        plan_generated_at_sec=plan_generated_at_sec,
        validated_at_sec=now_sec - 0.1,
        order_sha256=digest,
        source_plan_sha256="b" * 64,
    )


def _controller(**policy_overrides):
    policy = MissionControllerPolicy(
        qr_validation=QRValidationPolicy(
            max_observation_age_sec=1.0,
            expected_calibration_sha256=CALIBRATION_SHA256,
            expected_clock_id="unix_epoch",
        ),
        **policy_overrides,
    )
    return MissionController(registry=_registry(), robot_id="robot_1", policy=policy)


class MissionControllerTest(unittest.TestCase):
    def test_preserves_server_order_across_dispatch_arrival_and_retry(self):
        controller = _controller()
        first = controller.begin(
            initial_qr=_event("initial", "HOME", "station_HOME", "candidate_home", 99.8),
            server_task=_task(),
            now_sec=100.0,
        )

        self.assertEqual(first.station_id, "station_A")
        self.assertEqual(first.ordered_station_ids, ("station_A", "station_C", "station_B"))

        retried = controller.retry_current(
            dispatch_id=first.dispatch_id,
            reason="local obstacle replan",
            now_sec=100.2,
        )
        self.assertEqual(retried.station_id, "station_A")
        self.assertEqual(retried.ordered_station_ids, first.ordered_station_ids)
        self.assertEqual(retried.attempt_number, 2)

        second = controller.confirm_arrival(
            dispatch_id=retried.dispatch_id,
            arrival_qr=_event("arrival-a", "A", "station_A", "candidate_a", 100.3),
            now_sec=100.4,
        )
        self.assertIsNotNone(second)
        self.assertEqual(second.station_id, "station_C")
        self.assertEqual(second.station_index, 1)
        self.assertEqual(second.ordered_station_ids, first.ordered_station_ids)

        third = controller.confirm_arrival(
            dispatch_id=second.dispatch_id,
            arrival_qr=_event("arrival-c", "C", "station_C", "candidate_c", 100.5),
            now_sec=100.6,
        )
        self.assertIsNotNone(third)
        self.assertEqual(third.station_id, "station_B")

        completed = controller.confirm_arrival(
            dispatch_id=third.dispatch_id,
            arrival_qr=_event("arrival-b", "B", "station_B", "candidate_b", 100.7),
            now_sec=100.8,
        )
        self.assertIsNone(completed)
        self.assertEqual(controller.snapshot.state, MissionState.COMPLETED)

    def test_wrong_station_does_not_advance_and_replay_is_rejected(self):
        controller = _controller()
        first = controller.begin(
            initial_qr=_event("initial", "HOME", "station_HOME", "candidate_home", 99.8),
            server_task=_task(),
            now_sec=100.0,
        )
        wrong = _event("wrong-b", "B", "station_B", "candidate_b", 100.1)

        with self.assertRaisesRegex(MissionControlError, "wrong station"):
            controller.confirm_arrival(
                dispatch_id=first.dispatch_id,
                arrival_qr=wrong,
                now_sec=100.2,
            )
        self.assertEqual(controller.snapshot.current_station_id, "station_A")
        self.assertEqual(controller.snapshot.state, MissionState.NAVIGATING)
        with self.assertRaisesRegex(MissionControlError, "replayed"):
            controller.confirm_arrival(
                dispatch_id=first.dispatch_id,
                arrival_qr=wrong,
                now_sec=100.3,
            )
        self.assertEqual(controller.snapshot.current_station_id, "station_A")

        # Changing only the envelope ID must not make the same camera frames
        # usable as a new arrival observation.
        with self.assertRaisesRegex(MissionControlError, "consensus sample"):
            controller.confirm_arrival(
                dispatch_id=first.dispatch_id,
                arrival_qr=replace(wrong, event_id="wrong-b-new-envelope"),
                now_sec=100.3,
            )

    def test_retry_and_confirmation_budgets_fail_closed(self):
        retry_controller = _controller(max_attempts_per_station=2)
        retry_first = retry_controller.begin(
            initial_qr=_event("initial-retry", "HOME", "station_HOME", "candidate_home", 99.8),
            server_task=_task(),
            now_sec=100.0,
        )
        retry_second = retry_controller.retry_current(
            dispatch_id=retry_first.dispatch_id,
            reason="first failure",
            now_sec=100.1,
        )
        with self.assertRaisesRegex(MissionControlError, "budget exhausted"):
            retry_controller.retry_current(
                dispatch_id=retry_second.dispatch_id,
                reason="second failure",
                now_sec=100.2,
            )
        self.assertEqual(retry_controller.snapshot.state, MissionState.FAILED)
        self.assertIsNone(retry_controller.snapshot.active_dispatch)

        confirmation_controller = _controller(max_confirmation_rejections_per_station=2)
        confirmation_first = confirmation_controller.begin(
            initial_qr=_event("initial-confirm", "HOME", "station_HOME", "candidate_home", 99.8),
            server_task=_task(),
            now_sec=100.0,
        )
        with self.assertRaises(MissionControlError):
            confirmation_controller.confirm_arrival(
                dispatch_id=confirmation_first.dispatch_id,
                arrival_qr=_event("wrong-1", "B", "station_B", "candidate_b", 100.1),
                now_sec=100.2,
            )
        with self.assertRaises(MissionControlError):
            confirmation_controller.confirm_arrival(
                dispatch_id=confirmation_first.dispatch_id,
                arrival_qr=_event("wrong-2", "C", "station_C", "candidate_c", 100.3),
                now_sec=100.4,
            )
        self.assertEqual(confirmation_controller.snapshot.state, MissionState.FAILED)

    def test_rejects_stale_server_task_and_tampered_order(self):
        stale_controller = _controller(max_status_age_sec=5.0)
        with self.assertRaisesRegex(MissionControlError, "status is stale"):
            stale_controller.begin(
                initial_qr=_event("initial-stale", "HOME", "station_HOME", "candidate_home", 109.8),
                server_task=_task(now_sec=100.0),
                now_sec=110.0,
            )

        task = _task()
        tampered = ValidatedServerTask(
            **{
                **task.__dict__,
                "ordered_station_ids": ("station_A", "station_B", "station_C"),
            }
        )
        with self.assertRaisesRegex(MissionControlError, "order_sha256"):
            _controller().begin(
                initial_qr=_event("initial-tampered", "HOME", "station_HOME", "candidate_home", 99.8),
                server_task=tampered,
                now_sec=100.0,
            )

    def test_arrival_confirmation_must_postdate_active_dispatch(self):
        controller = _controller()
        first = controller.begin(
            initial_qr=_event("initial-time", "HOME", "station_HOME", "candidate_home", 99.8),
            server_task=_task(),
            now_sec=100.0,
        )
        with self.assertRaisesRegex(MissionControlError, "predates"):
            controller.confirm_arrival(
                dispatch_id=first.dispatch_id,
                arrival_qr=_event("old-arrival", "A", "station_A", "candidate_a", 99.7),
                now_sec=100.1,
            )
        self.assertEqual(controller.snapshot.current_station_id, "station_A")

    def test_rejects_completion_for_superseded_dispatch(self):
        controller = _controller()
        first = controller.begin(
            initial_qr=_event("initial-dispatch", "HOME", "station_HOME", "candidate_home", 99.8),
            server_task=_task(),
            now_sec=100.0,
        )
        retried = controller.retry_current(
            dispatch_id=first.dispatch_id,
            reason="route refresh",
            now_sec=100.1,
        )

        with self.assertRaisesRegex(MissionControlError, "does not match"):
            controller.confirm_arrival(
                dispatch_id=first.dispatch_id,
                arrival_qr=_event("arrival-old-dispatch", "A", "station_A", "candidate_a", 100.2),
                now_sec=100.3,
            )
        self.assertEqual(controller.snapshot.active_dispatch, retried)


if __name__ == "__main__":
    unittest.main()
