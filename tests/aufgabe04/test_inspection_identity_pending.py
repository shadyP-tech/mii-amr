import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.aufgabe04.artifacts.candidate_inspection_observation import (
    build_candidate_inspection_observation,
    load_candidate_inspection_observation,
)
from scripts.aufgabe04.real_robot.candidate.inspection_execution import (
    CandidateInspectionEffects,
    execute_candidate_inspection,
)
from scripts.aufgabe04.real_robot.candidate.inspection_route_search import (
    CandidateInspectionRouteUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)


PENDING_REASON = "measured_head_front_identity_unresolved"


def observation(**fields):
    return SimpleNamespace(**{
        "recommendation_path": None, "qr_observation_pose_path": None,
        "axis_observation_path": None, "inspection_observation_path": None,
        "qr_id": None, **fields})


class InspectionIdentityPendingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.frame = SimpleNamespace(normal=0.)
        self.run_count = 0

    def progress_observation(self, reason=PENDING_REASON, *, qr_id=None):
        path = self.root / (reason + ".json")
        payload = build_candidate_inspection_observation(
            candidate_uid="candidate", stream_id="camera", planning_frame="map",
            classification="unobservable", reasons=[reason],
            stand_center={"x_m": 1., "y_m": 2.},
            robot_pose={"x_m": 0., "y_m": 0., "yaw_rad": 0.},
            robot_profile_sha256="a" * 64, calibration_profile_sha256="b" * 64,
            stand_model_profile_sha256="c" * 64,
            sensor_stamps_sec=list(range(7)), sample_count=7,
            first_sensor_stamp_sec=0, sensor_stamp_sec=6,
            sample_gate_evidence={key: True for key in (
                "all_samples_stationary", "all_samples_synchronized",
                "all_samples_lidar_associated", "all_samples_fresh")},
            qr_id=qr_id, qr_sample_count=0 if qr_id is None else 2,
            camera_relative_yaw_rad=None, yaw_uncertainty_rad=None,
        )
        path.write_text(json.dumps(payload))
        return observation(inspection_observation_path=path)

    def resolved(self, *, joint=False):
        return observation(qr_id="QR_003", **{
            "recommendation_path" if joint else "qr_observation_pose_path": self.root / "resolved.json"})

    def unavailable(self, **status):
        return CandidateObservationUnavailableError(
            candidate_uid="candidate", observation_attempt_index=1,
            reason="unobservable", process_evidence={}, status_evidence=status)

    def forbidden(self, *args):
        self.fail("visible-head identity acquisition entered a motion/recovery path")

    def run_inspection(self, capture, *, max_views=8, **overrides):
        self.run_root = self.root / f"run_{self.run_count}"
        self.run_count += 1
        return execute_candidate_inspection(
            candidate_uid="candidate", candidate_root=self.run_root, initial_frame=self.frame,
            max_views=max_views, effects=CandidateInspectionEffects(**{
                "capture": capture, "canonical_normal": lambda frame: frame.normal,
                "progress_evidence": lambda frame, value: load_candidate_inspection_observation(
                    value.inspection_observation_path),
                "move_view": self.forbidden, "move_opposite": self.forbidden,
                "distance_recovery": self.forbidden, "move_distance_recovery": self.forbidden,
                "recover_lidar": self.forbidden, **overrides}))

    def assert_pending_exhaustion(self, failure, count):
        self.assertEqual(failure.reason, "candidate_local_inspection_exhausted")
        self.assertEqual(failure.process_evidence["inspection_exhaustion_reason"],
                         "head_visible_identity_pending_exhausted")
        progress = json.loads((self.run_root / "inspection_progress.json").read_text())
        self.assertEqual(progress["local_view_count"], count)
        self.assertFalse(progress["camera_distance_recovery_attempted"])
        self.assertEqual(progress["lidar_recovery_attempts"], [])
        self.assertEqual(progress["attempted_view_normals_rad"], [])
        return progress

    def test_one_passive_retry_uses_the_actual_stopped_frame_without_centering(self):
        pending = self.progress_observation()
        stopped = SimpleNamespace(normal=.1)
        calls = []

        def centered(frame, output, index):
            calls.append(("centered", frame, index))
            self.assertEqual(index, 0)
            return pending, stopped

        def capture(frame, output, index):
            calls.append(("passive", frame, index))
            return pending

        with self.assertRaises(CandidateObservationUnavailableError) as raised:
            self.run_inspection(capture, capture_centered=centered)
        self.assertEqual(calls, [("centered", self.frame, 0), ("passive", stopped, 1)])
        progress = self.assert_pending_exhaustion(raised.exception, 2)
        self.assertTrue(progress["identity_pending_passive_retry_used"])

    def test_second_capture_can_complete_identity_or_joint_recommendation(self):
        pending = self.progress_observation()
        for joint in (False, True):
            with self.subTest(joint=joint):
                resolved, calls = self.resolved(joint=joint), []

                def capture(frame, output, index):
                    calls.append((frame, index))
                    return pending if index == 0 else resolved

                result, frame = self.run_inspection(capture)
                self.assertIs(result, resolved)
                self.assertIs(frame, self.frame)
                self.assertEqual(calls, [(self.frame, 0), (self.frame, 1)])

    def test_generic_unavailable_retry_cannot_restore_motion(self):
        pending = self.progress_observation()
        for typed_failure in (False, True):
            with self.subTest(typed_failure=typed_failure):
                unavailable = self.progress_observation("current_head_unavailable")
                calls = []

                def capture(frame, output, index):
                    calls.append(index)
                    if index == 0:
                        return pending
                    if typed_failure:
                        raise self.unavailable(reason="current_head_unavailable")
                    return unavailable

                with self.assertRaises(CandidateObservationUnavailableError) as raised:
                    self.run_inspection(capture)
                self.assertEqual(calls, [0, 1])
                self.assert_pending_exhaustion(raised.exception, 2)

    def test_max_views_bounds_passive_retry(self):
        pending = self.progress_observation()
        for budget in (1, 2):
            with self.subTest(budget=budget):
                calls = []

                def capture(frame, output, index):
                    calls.append(index)
                    return pending

                with self.assertRaises(CandidateObservationUnavailableError) as raised:
                    self.run_inspection(capture, max_views=budget)
                self.assertEqual(calls, list(range(budget)))
                progress = self.assert_pending_exhaustion(raised.exception, budget)
                self.assertEqual(progress["identity_pending_passive_retry_used"], budget > 1)

    def test_certified_axis_from_retry_preserves_opposite_route_priority(self):
        pending, resolved = self.progress_observation(), self.resolved()
        axis = observation(axis_observation_path=self.root / "axis.json")
        opposite_frame, calls = SimpleNamespace(normal=3.14), []

        def capture(frame, output, index):
            calls.append(("capture", frame, index))
            return (pending, axis, resolved)[index]

        def opposite(frame, value, output, index):
            self.assertIs(value, axis)
            calls.append(("opposite", frame, index))
            return opposite_frame

        result, frame = self.run_inspection(capture, move_opposite=opposite)
        self.assertIs(result, resolved)
        self.assertIs(frame, opposite_frame)
        self.assertEqual(calls, [("capture", self.frame, 0), ("capture", self.frame, 1),
                                 ("opposite", self.frame, 2), ("capture", opposite_frame, 2)])

    def test_unavailable_certified_opposite_after_retry_does_not_fall_through(self):
        pending = self.progress_observation()
        axis = observation(axis_observation_path=self.root / "axis.json")

        def opposite(*args):
            raise CandidateInspectionRouteUnavailableError("blocked", reason_code="route_unavailable")

        with self.assertRaises(CandidateObservationUnavailableError) as raised:
            self.run_inspection(lambda frame, output, index: pending if index == 0 else axis,
                                move_opposite=opposite)
        self.assert_pending_exhaustion(raised.exception, 2)

    def test_certified_opposite_does_not_reset_the_extra_passive_retry_budget(self):
        pending = self.progress_observation()
        axis = observation(axis_observation_path=self.root / "axis.json")
        opposite_frame, captures = SimpleNamespace(normal=3.14), []

        def capture(frame, output, index):
            captures.append((frame, index))
            return axis if index == 1 else pending

        with self.assertRaises(CandidateObservationUnavailableError) as raised:
            self.run_inspection(capture, move_opposite=lambda *args: opposite_frame)
        self.assertEqual(captures, [(self.frame, 0), (self.frame, 1), (opposite_frame, 2)])
        self.assert_pending_exhaustion(raised.exception, 3)

    def test_raw_status_reason_without_validated_progress_does_not_latch(self):
        resolved, recoveries = self.resolved(), []

        def capture(frame, output, index):
            if index == 0:
                raise self.unavailable(reasons=[PENDING_REASON])
            return resolved

        def recover(frame, output, index):
            recoveries.append(index)
            return frame

        result, _ = self.run_inspection(capture, recover_lidar=recover,
                                       distance_recovery=lambda *args: None)
        self.assertIs(result, resolved)
        self.assertEqual(recoveries, [1])

    def test_provisional_identity_still_requires_current_face_identity(self):
        readable = self.progress_observation(qr_id="QR_003")
        resolved, captures = self.resolved(), []

        def capture(frame, output, index):
            captures.append((frame, index))
            return readable if index == 0 else resolved

        result, _ = self.run_inspection(capture)
        self.assertIs(result, resolved)
        self.assertEqual(captures, [(self.frame, 0), (self.frame, 1)])

    def test_terminal_error_from_passive_retry_propagates(self):
        pending = self.progress_observation()
        failure = RuntimeError("sensor lifecycle failed")

        def capture(frame, output, index):
            if index == 0:
                return pending
            raise failure

        with self.assertRaises(RuntimeError) as raised:
            self.run_inspection(capture)
        self.assertIs(raised.exception, failure)
        progress = json.loads((self.run_root / "inspection_progress.json").read_text())
        self.assertEqual(progress["termination_reason"], "observer_terminal_failure")

    def test_target_ineligibility_on_retry_preserves_immediate_deferral(self):
        pending = self.progress_observation()
        failure = CandidateObservationUnavailableError(
            candidate_uid="candidate", observation_attempt_index=1,
            reason="candidate_target_ineligible", process_evidence={},
            status_evidence={"state": "target_reconciliation_required"})

        def capture(frame, output, index):
            if index == 0:
                return pending
            raise failure

        with self.assertRaises(CandidateObservationUnavailableError) as raised:
            self.run_inspection(capture)
        self.assertIs(raised.exception, failure)
        progress = json.loads((self.run_root / "inspection_progress.json").read_text())
        self.assertEqual(progress["termination_reason"], "target_reconciliation_required")


if __name__ == "__main__":
    unittest.main()
