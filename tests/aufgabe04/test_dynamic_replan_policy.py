import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.dynamic_replan_policy import (  # noqa: E402
    DynamicReplanPolicy,
    DynamicReplanState,
)
from scripts.aufgabe04.navigation.models import Pose2D  # noqa: E402
from scripts.aufgabe04.navigation.viewpoint_recommendation import (  # noqa: E402
    MaterialTarget,
)


def target(
    *, x: float = 1.0, y: float = 2.0, yaw: float = 0.0,
    face_id: str = "face_a", evidence_state: str = "silhouette",
) -> MaterialTarget:
    return MaterialTarget(face_id, Pose2D(x, y, yaw), evidence_state)


class DynamicReplanPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = DynamicReplanPolicy(
            target_position_threshold_m=0.06,
            target_yaw_threshold_rad=math.radians(10.0),
            start_deviation_threshold_m=0.15,
            refresh_timeout_sec=3.0,
        )
        self.start = Pose2D(0.0, 0.0, 0.0)

    def first_planned_state(self):
        state, decision = self.policy.evaluate(
            DynamicReplanState(), target=target(), robot_pose=self.start, now_sec=1.0
        )
        self.assertTrue(decision.should_replan)
        self.assertEqual(state.target_revision, 1)
        return self.policy.mark_route_planned(
            state, planned_start=self.start, now_sec=1.0
        )

    def test_tiny_changes_do_not_increment_or_replan(self):
        state = self.first_planned_state()
        state, decision = self.policy.evaluate(
            state,
            target=target(x=1.02, yaw=math.radians(3.0)),
            robot_pose=Pose2D(0.03, 0.0),
            now_sec=1.5,
        )
        self.assertEqual(state.target_revision, 1)
        self.assertFalse(decision.target_changed)
        self.assertFalse(decision.should_replan)

    def test_hysteresis_is_anchored_to_last_material_target(self):
        state = self.first_planned_state()
        for index, x in enumerate((1.02, 1.04), start=1):
            state, decision = self.policy.evaluate(
                state, target=target(x=x), robot_pose=self.start, now_sec=1.0 + index * 0.2
            )
            self.assertFalse(decision.target_changed)
        state, decision = self.policy.evaluate(
            state, target=target(x=1.061), robot_pose=self.start, now_sec=1.8
        )
        self.assertTrue(decision.target_changed)
        self.assertEqual(state.target_revision, 2)
        self.assertIn("target_revision_changed", decision.reasons)

    def test_wrap_safe_yaw_and_material_face_or_evidence(self):
        initial = target(yaw=math.pi - 0.02)
        state, _ = self.policy.evaluate(
            DynamicReplanState(), target=initial, robot_pose=self.start, now_sec=1.0
        )
        state = self.policy.mark_route_planned(state, planned_start=self.start, now_sec=1.0)
        state, wrapped = self.policy.evaluate(
            state,
            target=target(yaw=-math.pi + 0.02),
            robot_pose=self.start,
            now_sec=1.2,
        )
        self.assertFalse(wrapped.target_changed)
        state, face = self.policy.evaluate(
            state, target=target(face_id="face_b"), robot_pose=self.start, now_sec=1.4
        )
        self.assertTrue(face.target_changed)
        self.assertEqual(state.target_revision, 2)
        state = self.policy.mark_route_planned(state, planned_start=self.start, now_sec=1.4)
        state, evidence = self.policy.evaluate(
            state,
            target=target(face_id="face_b", evidence_state="hard_qr"),
            robot_pose=self.start,
            now_sec=1.6,
        )
        self.assertTrue(evidence.target_changed)
        self.assertEqual(state.target_revision, 3)

    def test_material_start_deviation_and_refresh(self):
        state = self.first_planned_state()
        state, moved = self.policy.evaluate(
            state, target=target(), robot_pose=Pose2D(0.16, 0.0), now_sec=2.0
        )
        self.assertIn("material_start_deviation", moved.reasons)
        state = self.policy.mark_route_planned(
            state, planned_start=Pose2D(0.16, 0.0), now_sec=2.0
        )
        state, refresh = self.policy.evaluate(
            state, target=target(), robot_pose=Pose2D(0.17, 0.0), now_sec=5.0
        )
        self.assertIn("refresh_timeout", refresh.reasons)

    def test_target_locked_transit_uses_heartbeat_without_motion_replan(self):
        policy = DynamicReplanPolicy(
            start_deviation_threshold_m=0.15,
            refresh_timeout_sec=3.0,
            replan_on_start_deviation=False,
        )
        state, first = policy.evaluate(
            DynamicReplanState(), target=target(), robot_pose=self.start, now_sec=1.0
        )
        self.assertTrue(first.should_replan)
        state = policy.mark_route_planned(state, planned_start=self.start, now_sec=1.0)

        state, moving = policy.evaluate(
            state, target=target(), robot_pose=Pose2D(0.50, 0.0), now_sec=2.0
        )
        self.assertFalse(moving.should_replan)
        self.assertNotIn("material_start_deviation", moving.reasons)

        state, heartbeat = policy.evaluate(
            state, target=target(), robot_pose=Pose2D(0.70, 0.0), now_sec=4.0
        )
        self.assertTrue(heartbeat.should_replan)
        self.assertEqual(heartbeat.reasons, ("refresh_timeout",))

    def test_terminal_corridor_suppresses_start_replan_but_emits_heartbeat(self):
        state = self.first_planned_state()
        state, before_deadline = self.policy.evaluate(
            state,
            target=target(),
            robot_pose=Pose2D(0.70, 2.0, 0.0),
            now_sec=2.0,
        )
        self.assertFalse(before_deadline.should_replan)
        self.assertEqual(before_deadline.reasons, ())

        state, decision = self.policy.evaluate(
            state,
            target=target(),
            robot_pose=Pose2D(0.70, 2.0, 0.0),
            now_sec=5.0,
        )
        self.assertFalse(decision.target_changed)
        self.assertTrue(decision.should_replan)
        self.assertNotIn("material_start_deviation", decision.reasons)
        self.assertEqual(decision.reasons, ("refresh_timeout",))

    def test_material_target_change_overrides_terminal_corridor_lock(self):
        state = self.first_planned_state()
        state, decision = self.policy.evaluate(
            state,
            target=target(x=1.08),
            robot_pose=Pose2D(0.70, 2.0, 0.0),
            now_sec=5.0,
        )
        self.assertTrue(decision.target_changed)
        self.assertTrue(decision.should_replan)
        self.assertIn("target_revision_changed", decision.reasons)

    def test_time_rollback_fails_closed_until_explicit_reset(self):
        state = self.first_planned_state()
        state, rollback = self.policy.evaluate(
            state, target=target(), robot_pose=self.start, now_sec=0.5
        )
        self.assertTrue(rollback.fail_closed)
        self.assertFalse(rollback.should_replan)
        self.assertEqual(rollback.reasons, ("time_moved_backwards",))
        state, still_closed = self.policy.evaluate(
            state, target=target(), robot_pose=self.start, now_sec=2.0
        )
        self.assertTrue(still_closed.fail_closed)
        reset = self.policy.reset_after_clock_change(state, now_sec=0.0)
        reset, decision = self.policy.evaluate(
            reset, target=target(), robot_pose=self.start, now_sec=0.1
        )
        self.assertFalse(decision.fail_closed)
        self.assertTrue(decision.should_replan)


if __name__ == "__main__":
    unittest.main()
