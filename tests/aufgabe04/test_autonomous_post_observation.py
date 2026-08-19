from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.real_robot import autonomous_post_observation as policy
from scripts.aufgabe04.real_robot.autonomous_post_observation import (
    DYNAMIC_MAP_TO_ODOM_UNAVAILABLE,
    PostObservationLocalizationConfig,
    PostObservationLocalizationEffects,
    PostObservationLocalizationError,
    admit_post_observation_localization,
    evaluate_post_observation_localization_retry,
)


def dynamic_gap_evidence() -> dict[str, object]:
    return {
        "ok": False,
        "failures": [DYNAMIC_MAP_TO_ODOM_UNAVAILABLE],
        "observations": [
            {"name": "scan freshness", "ok": True, "detail": "fresh", "data": {}},
            {
                "name": "localization transform ownership",
                "ok": False,
                "detail": DYNAMIC_MAP_TO_ODOM_UNAVAILABLE,
                "data": {
                    "localization_source": "amcl",
                    "execution_pose_owner": "amcl",
                    "amcl_fresh": True,
                    "route_transform_fresh": True,
                    "map_to_odom_dynamic_fresh": False,
                    "external_tf_owner_candidates": [],
                    "ambiguous_owner_evidence": [],
                    "map_to_odom_dynamic": {
                        "available": False,
                        "dynamic": False,
                    }
                },
            },
            {"name": "cmd_vel ownership", "ok": True, "detail": "ok", "data": {}},
        ],
    }


def config(maximum: int = 2) -> PostObservationLocalizationConfig:
    return PostObservationLocalizationConfig(
        session_root=Path("session"),
        session_id="mission",
        recorded_viewpoint_id="survey_vp_005",
        maximum_retry_count=maximum,
    )


class AutonomousPostObservationTest(unittest.TestCase):
    def test_exact_dynamic_gap_is_the_only_retryable_evidence(self):
        self.assertTrue(
            evaluate_post_observation_localization_retry(
                dynamic_gap_evidence()
            ).retryable
        )
        mutations = (
            {"ok": True},
            {"failures": ["scan stale"]},
            {
                "failures": [
                    DYNAMIC_MAP_TO_ODOM_UNAVAILABLE,
                    "scan stale",
                ]
            },
            {
                "observations": [
                    *dynamic_gap_evidence()["observations"],
                    {"name": "scan freshness", "ok": False},
                ]
            },
            {
                "observations": [
                    *dynamic_gap_evidence()["observations"],
                    "malformed",
                ]
            },
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                evidence = {**dynamic_gap_evidence(), **mutation}
                self.assertFalse(
                    evaluate_post_observation_localization_retry(
                        evidence
                    ).retryable
                )

    def test_ownership_conflict_or_stale_pose_is_not_retryable(self):
        for field, value in (
            ("amcl_fresh", False),
            ("route_transform_fresh", False),
            ("external_tf_owner_candidates", ["/slam_toolbox"]),
            ("ambiguous_owner_evidence", ["multiple owners"]),
        ):
            with self.subTest(field=field):
                evidence = dynamic_gap_evidence()
                ownership = evidence["observations"][1]
                ownership["data"][field] = value
                decision = evaluate_post_observation_localization_retry(
                    evidence
                )
                self.assertFalse(decision.retryable)
                self.assertEqual(
                    decision.reason,
                    "ownership_observation_not_exact_dynamic_gap",
                )

    def test_retries_with_unique_evidence_then_admits_without_motion_authority(self):
        attempts: list[Path] = []
        events: list[dict[str, object]] = []

        def admit(path: Path) -> Pose2D:
            attempts.append(path)
            if len(attempts) == 1:
                raise RuntimeError(DYNAMIC_MAP_TO_ODOM_UNAVAILABLE)
            return Pose2D(0.9, -0.02, -0.01)

        outcome = admit_post_observation_localization(
            config(),
            PostObservationLocalizationEffects(
                admit_localization=admit,
                read_evidence=lambda path: dynamic_gap_evidence(),
                event_sink=lambda event: events.append(dict(event)),
                clock=lambda: 42.0,
            ),
        )

        self.assertEqual(outcome.retry_count, 1)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(str(attempts[0]).endswith("post_observation_localization.json"))
        self.assertTrue(
            str(attempts[1]).endswith(
                "post_observation_localization_retry_001.json"
            )
        )
        self.assertEqual(
            [event["event"] for event in events],
            [
                "post_observation_localization_retry_scheduled",
                "post_observation_localization_admitted",
            ],
        )
        self.assertTrue(all(event["motion_authorized"] is False for event in events))
        self.assertTrue(
            all(event["additional_typed_run_required"] is False for event in events)
        )

    def test_retry_exhaustion_is_structured_and_attempt_count_is_bounded(self):
        attempts: list[Path] = []

        def reject(path: Path) -> Pose2D:
            attempts.append(path)
            raise RuntimeError(DYNAMIC_MAP_TO_ODOM_UNAVAILABLE)

        with self.assertRaises(PostObservationLocalizationError) as raised:
            admit_post_observation_localization(
                config(maximum=2),
                PostObservationLocalizationEffects(
                    admit_localization=reject,
                    read_evidence=lambda path: dynamic_gap_evidence(),
                ),
            )

        self.assertEqual(len(attempts), 3)
        fields = raised.exception.to_failure_fields()
        self.assertEqual(fields["recorded_viewpoint_id"], "survey_vp_005")
        self.assertEqual(fields["post_observation_localization_retry_count"], 2)
        self.assertFalse(fields["post_observation_retry_motion_authorized"])
        self.assertFalse(fields["post_observation_retry_motion_published"])
        self.assertFalse(fields["additional_typed_run_required"])
        self.assertEqual(
            len(fields["post_observation_localization_evidence"]),
            3,
        )

    def test_unrelated_failure_never_retries(self):
        attempts = 0

        def reject(path: Path) -> Pose2D:
            nonlocal attempts
            del path
            attempts += 1
            raise RuntimeError("scan stale")

        with self.assertRaises(PostObservationLocalizationError) as raised:
            admit_post_observation_localization(
                config(),
                PostObservationLocalizationEffects(
                    admit_localization=reject,
                    read_evidence=lambda path: {
                        "ok": False,
                        "failures": ["scan stale"],
                        "observations": [],
                    },
                ),
            )
        self.assertEqual(attempts, 1)
        self.assertEqual(
            raised.exception.reason_code,
            "failure_set_not_exact_dynamic_map_to_odom_gap",
        )

    def test_module_has_no_prompt_process_or_ros_import(self):
        source = inspect.getsource(policy)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        self.assertFalse(any(name in {"rclpy", "subprocess"} for name in imported))
        self.assertNotIn("input(", source)
        self.assertNotIn("run_autonomous_stand_exploration", source)


if __name__ == "__main__":
    unittest.main()
