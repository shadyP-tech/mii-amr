from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from scripts.aufgabe04.navigation.approach.candidate_frame_reprojection import (
    CandidateFrameProvenance,
    reproject_candidate_point,
)
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.localization.odom_execution_certificate import (
    map_pose_to_odom,
    odom_pose_to_map,
)
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


def _tf_data(
    target: str, source: str, pose: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "available": True,
        "target_frame": target,
        "source_frame": source,
        "observed_target_frame": target,
        "observed_source_frame": source,
        "stamp_sec": 100.02,
        "capture_time_sec": 100.03,
        "age_sec": 0.01,
        "x_m": pose[0], "y_m": pose[1], "yaw_rad": pose[2],
    }


def _preflight(
    *,
    explicit_requirement: bool = True,
    samples: list[dict[str, object]] | None = None,
    include_direct_tf: bool = True,
) -> RosPreflightResult:
    direct_transform = _tf_data("map", "odom", (1.0, -0.5, 0.2))
    odom_transform = _tf_data("odom", "base_footprint", (0.2, -0.1, 0.3))
    observations = [
        RosObservation(
            "stationary map<-odom transform samples",
            True,
            "paired_samples=2/2",
            {"required_pair_count": 2},
        ),
        RosObservation("odom freshness", True, "fresh", {}),
        RosObservation("tf odom->base_footprint", True, "fresh", odom_transform),
    ]
    if include_direct_tf:
        observations.append(
            RosObservation(
                "tf map->odom",
                True,
                "age=0.010s",
                deepcopy(direct_transform),
            )
        )
    return RosPreflightResult(
        ok=True,
        failures=[],
        observations=observations,
        runtime_config={
            "localization_source": "amcl", "base_frame": "base_footprint",
        },
        preflight_requirements={
            "stationary_map_from_odom_pairing_requested": (
                explicit_requirement
            ),
            "stationary_map_from_odom_pairing_required": True,
        },
        odom_pose={
            "frame_id": "odom", "child_frame_id": "base_footprint",
            "x_m": 0.2, "y_m": -0.1, "yaw_rad": 0.3,
        },
        map_from_odom=direct_transform,
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
        preflight.observations[-1].data["yaw_rad"] = -3.13

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

        self.assertEqual(
            frame.current_pose,
            odom_pose_to_map(Pose2D(0.2, -0.1, 0.3), frame.map_from_odom),
        )
        self.assertAlmostEqual(frame.current_pose.x_m, 1.21588024864775)
        self.assertAlmostEqual(frame.current_pose.y_m, -0.558272791625112)
        self.assertAlmostEqual(frame.current_pose.yaw_rad, 0.5)
        self.assertEqual(frame.map_frame, "map")
        self.assertAlmostEqual(frame.map_from_odom.x_m, 1.0)

    def test_missing_odom_never_falls_back_to_chained_map_pose(self):
        with self.assertRaisesRegex(RuntimeError, "no odom pose"):
            self._build(replace(_preflight(), odom_pose=None))

    def test_odom_requires_matching_pose_and_capture_frames(self):
        mutations = (
            ("pose", "frame_id", "other_odom"),
            ("pose", "child_frame_id", "other_base"),
            ("capture", "target_frame", "other_odom"),
            ("capture", "source_frame", "other_base"),
            ("capture", "observed_target_frame", "other_odom"),
            ("capture", "observed_source_frame", "other_base"),
        )
        for target, key, value in mutations:
            with self.subTest(target=target, key=key):
                preflight = _preflight()
                data = (
                    preflight.odom_pose if target == "pose"
                    else preflight.observations[2].data
                )
                data[key] = value
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    self._build(preflight)

    def test_odom_requires_finite_pose_and_capture_time(self):
        for target, key in (
            ("pose", "x_m"), ("pose", "y_m"), ("pose", "yaw_rad"),
            ("capture", "stamp_sec"), ("capture", "capture_time_sec"),
        ):
            for value in (float("nan"), float("inf"), True, None):
                with self.subTest(target=target, key=key, value=value):
                    preflight = _preflight()
                    data = (
                        preflight.odom_pose if target == "pose"
                        else preflight.observations[2].data
                    )
                    data[key] = value
                    with self.assertRaises(RuntimeError):
                        self._build(preflight)

    def test_odom_requires_both_successful_freshness_observations(self):
        for name in ("odom freshness", "tf odom->base_footprint"):
            for mutation in ("missing", "failed", "duplicated"):
                with self.subTest(name=name, mutation=mutation):
                    preflight = _preflight()
                    observation = next(o for o in preflight.observations if o.name == name)
                    if mutation == "missing":
                        preflight.observations.remove(observation)
                    elif mutation == "failed":
                        index = preflight.observations.index(observation)
                        preflight.observations[index] = replace(observation, ok=False)
                    else:
                        preflight.observations.append(observation)
                    with self.assertRaisesRegex(RuntimeError, "successful.*" + name):
                        self._build(preflight)

    def test_odom_pose_must_come_from_the_successful_tf_capture(self):
        preflight = _preflight()
        preflight.odom_pose["x_m"] += 0.001
        with self.assertRaisesRegex(RuntimeError, "odom pose capture mismatch"):
            self._build(preflight)

    def test_direct_transform_must_match_the_admitted_capture(self):
        for key, value in (
            ("x_m", 1.001), ("capture_time_sec", 100.04),
            ("observed_source_frame", "wrong_odom"),
        ):
            with self.subTest(key=key):
                preflight = _preflight()
                preflight.map_from_odom[key] = value
                with self.assertRaisesRegex(RuntimeError, "direct TF capture mismatch"):
                    self._build(preflight)

    def test_base_frame_identity_is_required(self):
        preflight = _preflight()
        preflight.runtime_config.pop("base_frame")
        with self.assertRaisesRegex(RuntimeError, "base frame identity"):
            self._build(preflight)

    def test_projection_evidence_preserves_complete_detached_capture_metadata(self):
        preflight = _preflight()
        preflight.map_from_odom["quaternion"] = {"norm": 1.0}
        preflight.observations[-1].data["quaternion"] = {"norm": 1.0}
        frame = self._build(preflight)
        evidence = frame.to_evidence()
        provenance = evidence["pose_provenance"]
        self.assertEqual(
            provenance["pose_basis"],
            "direct_map_from_odom_times_observed_odom_pose",
        )
        self.assertEqual(provenance["map_from_odom_capture"], preflight.map_from_odom)
        self.assertEqual(
            provenance["odom_pose_capture"], preflight.observations[2].data,
        )
        self.assertEqual(
            provenance["diagnostic_chained_map_pose"],
            {"x_m": 0.2, "y_m": -0.1, "yaw_rad": 0.3},
        )
        self.assertAlmostEqual(
            provenance["chained_to_authoritative_translation_delta_m"],
            math.hypot(frame.current_pose.x_m - 0.2, frame.current_pose.y_m + 0.1),
        )
        provenance["map_from_odom_capture"]["quaternion"]["norm"] = 2.0
        self.assertEqual(
            frame.to_evidence()["pose_provenance"]["map_from_odom_capture"]["quaternion"],
            {"norm": 1.0},
        )
        preflight.map_from_odom["quaternion"]["norm"] = 3.0
        self.assertEqual(
            frame.to_evidence()["pose_provenance"]["map_from_odom_capture"]["quaternion"],
            {"norm": 1.0},
        )

    def test_recorded_selection_uses_same_transform_for_anchor_and_candidate(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures" /
             "candidate_planning_frame_20260911T124500Z.json").read_text()
        )
        payload = fixture["preflight"]
        preflight = RosPreflightResult(
            **{key: value for key, value in payload.items() if key != "observations"},
            observations=[RosObservation(**value) for value in payload["observations"]],
        )
        chained = Pose2D(**{
            key: preflight.route_pose[key] for key in ("x_m", "y_m", "yaw_rad")
        })
        frame = build_candidate_planning_frame(
            preflight, current_pose=chained, map_frame="map", odom_frame="odom",
        )
        self.assertEqual(frame.current_pose, Pose2D(**fixture["expected_authoritative_pose"]))
        self.assertNotEqual(frame.current_pose, chained)
        self.assertAlmostEqual(
            frame.pose_provenance["chained_to_authoritative_translation_delta_m"],
            fixture["expected_chained_delta_m"],
        )
        anchor_odom = map_pose_to_odom(frame.current_pose, frame.map_from_odom)
        self.assertAlmostEqual(anchor_odom.x_m, preflight.odom_pose["x_m"])
        self.assertAlmostEqual(anchor_odom.y_m, preflight.odom_pose["y_m"])
        self.assertAlmostEqual(anchor_odom.yaw_rad, preflight.odom_pose["yaw_rad"])
        recorded_projection = fixture["recorded_candidate_projection"]
        projection = reproject_candidate_point(
            CandidateFrameProvenance.from_mapping(recorded_projection["provenance"]),
            frame.map_from_odom,
        )
        self.assertEqual(
            projection.current_map_point.to_mapping(),
            recorded_projection["current_map_point"],
        )
        self.assertAlmostEqual(
            math.hypot(
                projection.current_map_point.x_m - frame.current_pose.x_m,
                projection.current_map_point.y_m - frame.current_pose.y_m,
            ),
            math.hypot(
                projection.canonical_odom_point.x_m - anchor_odom.x_m,
                projection.canonical_odom_point.y_m - anchor_odom.y_m,
            ),
        )
        self.assertEqual(
            frame.to_evidence()["pose_provenance"]["map_from_odom_capture"],
            preflight.map_from_odom,
        )
        self.assertEqual(
            frame.to_evidence()["pose_provenance"]["diagnostic_chained_map_capture"],
            next(o.data for o in preflight.observations if o.name == "tf map->base_footprint"),
        )
    def _build(self, preflight):
        return build_candidate_planning_frame(
            preflight,
            current_pose=Pose2D(0.2, -0.1, 0.3),
            map_frame="map",
            odom_frame="odom",
        )


if __name__ == "__main__":
    unittest.main()
