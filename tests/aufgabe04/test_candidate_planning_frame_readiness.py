from copy import deepcopy
import unittest

from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.ros_preflight import (
    RosObservation,
    RosPreflightRequirements,
    RosPreflightResult,
)
from scripts.aufgabe04.real_robot.readiness.candidate_planning_frame import (
    CANDIDATE_PLANNING_FRAME_PREFLIGHT_REQUIREMENTS,
    build_candidate_planning_frame,
)


def _sample(index: int) -> dict[str, object]:
    return {
        "source": "direct_dynamic_tf",
        "target_frame": "map",
        "source_frame": "odom",
        "amcl_sample_index": index,
        "stamp_nanoseconds": 100_000_000_000 + index * 10_000_000,
        "receipt_time_nanoseconds": 200_000_000_000 + index * 10_000_000,
        "x_m": 1.0,
        "y_m": -0.5,
        "yaw_rad": 0.2,
    }


def _preflight(
    *,
    explicit_requirement: bool = True,
    samples: list[dict[str, object]] | None = None,
    include_direct_tf: bool = True,
) -> RosPreflightResult:
    observations = [
        RosObservation(
            "stationary map<-odom transform samples",
            True,
            "paired_samples=2/2",
            {"required_pair_count": 2},
        )
    ]
    if include_direct_tf:
        observations.append(
            RosObservation(
                "tf map->odom",
                True,
                "age=0.010s",
                {"target_frame": "map", "source_frame": "odom"},
            )
        )
    return RosPreflightResult(
        ok=True,
        failures=[],
        observations=observations,
        runtime_config={"localization_source": "amcl"},
        preflight_requirements={
            "stationary_map_from_odom_pairing_requested": (
                explicit_requirement
            ),
            "stationary_map_from_odom_pairing_required": True,
        },
        map_from_odom={
            "target_frame": "map",
            "source_frame": "odom",
            "stamp_sec": 100.02,
            "x_m": 1.0,
            "y_m": -0.5,
            "yaw_rad": 0.2,
        },
        stationary_map_from_odom_samples=(
            [_sample(0), _sample(1)] if samples is None else samples
        ),
    )


class CandidatePlanningFrameReadinessTests(unittest.TestCase):
    def test_pairing_policy_is_independent_of_execution_pose_owner(self):
        ordinary = RosPreflightRequirements()
        candidate = CANDIDATE_PLANNING_FRAME_PREFLIGHT_REQUIREMENTS

        self.assertFalse(
            ordinary.stationary_map_from_odom_pairing_required(
                execution_pose_owner="",
            )
        )
        self.assertTrue(
            ordinary.stationary_map_from_odom_pairing_required(
                execution_pose_owner="odom",
            )
        )
        self.assertTrue(
            candidate.stationary_map_from_odom_pairing_required(
                execution_pose_owner="",
            )
        )
        self.assertEqual(
            candidate.to_evidence(execution_pose_owner=""),
            {
                "stationary_map_from_odom_pairing_requested": True,
                "stationary_map_from_odom_pairing_required": True,
            },
        )

    def test_requirement_rejects_non_boolean_configuration(self):
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            RosPreflightRequirements(  # type: ignore[arg-type]
                require_stationary_map_from_odom_pairing="yes"
            )

    def test_candidate_frame_requires_explicit_pairing_evidence(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "did not explicitly require",
        ):
            build_candidate_planning_frame(
                _preflight(explicit_requirement=False),
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_incomplete_pair_window(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            build_candidate_planning_frame(
                _preflight(samples=[_sample(0)]),
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_requires_fresh_direct_tf_observation(self):
        with self.assertRaisesRegex(RuntimeError, "tf map->odom"):
            build_candidate_planning_frame(
                _preflight(include_direct_tf=False),
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_final_translation_tamper(self):
        preflight = _preflight()
        assert preflight.map_from_odom is not None
        preflight.map_from_odom["x_m"] = 1.031

        with self.assertRaisesRegex(RuntimeError, "translation delta"):
            build_candidate_planning_frame(
                preflight,
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_final_yaw_tamper(self):
        preflight = _preflight()
        assert preflight.map_from_odom is not None
        preflight.map_from_odom["yaw_rad"] = 0.231

        with self.assertRaisesRegex(RuntimeError, "yaw delta"):
            build_candidate_planning_frame(
                preflight,
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_stale_final_transform_stamp(self):
        preflight = _preflight()
        assert preflight.map_from_odom is not None
        preflight.map_from_odom["stamp_sec"] = 100.005

        with self.assertRaisesRegex(RuntimeError, "older than"):
            build_candidate_planning_frame(
                preflight,
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_missing_final_transform_stamp(self):
        preflight = _preflight()
        assert preflight.map_from_odom is not None
        preflight.map_from_odom.pop("stamp_sec")

        with self.assertRaisesRegex(RuntimeError, "timestamp is missing"):
            build_candidate_planning_frame(
                preflight,
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_non_finite_final_transform(self):
        preflight = _preflight()
        assert preflight.map_from_odom is not None
        preflight.map_from_odom["x_m"] = float("nan")

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            build_candidate_planning_frame(
                preflight,
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_rejects_non_finite_last_paired_transform(self):
        samples = [_sample(0), _sample(1)]
        samples[-1]["yaw_rad"] = float("inf")

        with self.assertRaisesRegex(RuntimeError, "last paired.*non-finite"):
            build_candidate_planning_frame(
                _preflight(samples=samples),
                current_pose=Pose2D(0.0, 0.0, 0.0),
                map_frame="map",
                odom_frame="odom",
            )

    def test_candidate_frame_accepts_bounded_wrapped_yaw_delta(self):
        samples = [_sample(0), _sample(1)]
        samples[-1]["yaw_rad"] = 3.13
        preflight = _preflight(samples=deepcopy(samples))
        assert preflight.map_from_odom is not None
        preflight.map_from_odom["yaw_rad"] = -3.13

        frame = build_candidate_planning_frame(
            preflight,
            current_pose=Pose2D(0.0, 0.0, 0.0),
            map_frame="map",
            odom_frame="odom",
        )

        self.assertAlmostEqual(frame.map_from_odom.yaw_rad, -3.13)

    def test_candidate_frame_accepts_complete_independent_requirement(self):
        frame = build_candidate_planning_frame(
            _preflight(),
            current_pose=Pose2D(0.2, -0.1, 0.3),
            map_frame="map",
            odom_frame="odom",
        )

        self.assertEqual(frame.current_pose, Pose2D(0.2, -0.1, 0.3))
        self.assertEqual(frame.map_frame, "map")
        self.assertAlmostEqual(frame.map_from_odom.x_m, 1.0)


if __name__ == "__main__":
    unittest.main()
