from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.missions.plan_first_detected_station import (  # noqa: E402
    build_parser as build_first_detected_station_parser,
    load_and_validate_confirmation_receipt,
    main as plan_first_detected_station_main,
    validate_observation_provenance,
)
from scripts.aufgabe04.navigation.missions.plan_detected_stand_exploration import (  # noqa: E402
    build_parser as build_detected_stand_exploration_parser,
    main as plan_detected_stand_exploration_main,
    start_pose_from_args,
)
from scripts.aufgabe04.navigation.missions import (  # noqa: E402
    plan_detected_stand_exploration as exploration_planner,
)
from scripts.aufgabe04.navigation.approach.create_detected_station_confirmation import (  # noqa: E402
    build_parser as build_detected_station_confirmation_parser,
    main as create_detected_station_confirmation_main,
)
from scripts.aufgabe04.navigation.execution.route_context import file_sha256  # noqa: E402
from scripts.aufgabe04.navigation.planning.map_io import freeze_map_bundle  # noqa: E402
from scripts.aufgabe04.perception.models import StandCandidate  # noqa: E402
from scripts.aufgabe04.perception.stand_confirmation import (  # noqa: E402
    StandConfirmationAccumulator,
    StandConfirmationConfig,
    select_confirmed_stand_by_id,
    select_first_confirmed_stand,
    select_unique_confirmed_stand,
)
from scripts.aufgabe04.perception.stand_observation import (  # noqa: E402
    DEFAULT_OBSERVATION_TIMING_LIMITS,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVER_CLOCK_ROS_SYSTEM_TIME,
    RUNTIME_TIMING_LIMITS_KEY,
    TF_LOOKUP_MODE_SCAN_TIME_EXACT,
    ObservationProvenance,
    PlanarTransform,
    observation_from_candidate,
    observation_from_payload,
    observation_to_payload,
    load_observation_jsonl_snapshot,
    validated_observation_stream_clock,
    write_observation_jsonl,
)
from scripts.aufgabe04.stations.detected_station_layout import (  # noqa: E402
    DetectedStationLayoutConfig,
    station_from_confirmed_stand,
)


def write_free_map(root: Path, *, width=30, height=30, resolution=0.1) -> Path:
    (root / "map.pgm").write_text(
        f"P2\n{width} {height}\n255\n" + " ".join(["255"] * width * height) + "\n"
    )
    (root / "map.yaml").write_text(
        "\n".join(
            [
                "image: map.pgm",
                f"resolution: {resolution}",
                "origin: [-1.0, -1.0, 0.0]",
                "negate: 0",
                "occupied_thresh: 0.65",
                "free_thresh: 0.20",
                "mode: trinary",
            ]
        )
        + "\n"
    )
    return root / "map.yaml"


def provenance(
    *,
    map_yaml: Path | None = None,
    tf_age_sec=0.1,
    map_frame="map",
    observer_clock=OBSERVER_CLOCK_ROS_SYSTEM_TIME,
    use_sim_time=False,
    observer_clock_sec=10.1,
    scan_stamp_sec=10.0,
):
    map_bundle = (
        None
        if map_yaml is None
        else freeze_map_bundle(
            map_yaml,
            semantic_map_id=map_yaml.stem,
            planning_frame=map_frame,
        )
    )
    tf_stamp_sec = observer_clock_sec - tf_age_sec
    return ObservationProvenance(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        observer_version="test-observer",
        resolved_scan_topic="/scan",
        scan_frame="base_scan",
        map_frame=map_frame,
        base_frame="base_footprint",
        localization_source="amcl",
        scan_stamp_sec=scan_stamp_sec,
        tf_lookup_stamp_sec=tf_stamp_sec,
        tf_age_sec=tf_age_sec,
        runtime_config={
            "scan_topic": "/scan",
            "use_sim_time": use_sim_time,
            RUNTIME_TIMING_LIMITS_KEY: (
                DEFAULT_OBSERVATION_TIMING_LIMITS.as_dict()
            ),
        },
        observer_clock=observer_clock,
        observer_clock_sec=observer_clock_sec,
        scan_age_sec=observer_clock_sec - scan_stamp_sec,
        tf_scan_skew_sec=abs(tf_stamp_sec - scan_stamp_sec),
        tf_query_stamp_sec=scan_stamp_sec,
        tf_lookup_mode=TF_LOOKUP_MODE_SCAN_TIME_EXACT,
        map_yaml=str(map_yaml or ""),
        map_yaml_sha256=file_sha256(map_yaml) if map_yaml else "",
        map_image_sha256="" if map_bundle is None else map_bundle.image_sha256,
        map_bundle_sha256="" if map_bundle is None else map_bundle.bundle_sha256,
    )


def candidate(candidate_id="candidate_1"):
    return StandCandidate(
        candidate_id=candidate_id,
        bearing_rad=0.0,
        distance_m=1.0,
        approximate_width_m=0.12,
        center_x_m=1.0,
        center_y_m=0.0,
        point_count=4,
        confidence=0.8,
    )


def observation(index: int, *, x=0.5, y=0.5, observed_at=10.0, map_yaml=None):
    base = observation_from_candidate(
        candidate(f"candidate_{index}"),
        transform_scan_to_map=PlanarTransform(x - 1.0, y, 0.0),
        observed_at_sec=observed_at,
        provenance=provenance(map_yaml=map_yaml),
        observation_index=index,
    )
    return base


def write_confirmation(path: Path, *, stand_id="detected_stand_01", station_id="A"):
    path.write_text(
        json.dumps(
            {
                "confirmation_source": "operator",
                "operator_confirmed": True,
                "stand_id": stand_id,
                "station_id": station_id,
            }
        )
    )


class DetectedStandPlannerStartPoseTest(unittest.TestCase):
    def test_tf_discovery_timeout_uses_monotonic_time_not_sim_clock(self):
        class FakeTransformException(Exception):
            pass

        class FakeRclpy:
            @staticmethod
            def init(args=None):
                return None

            @staticmethod
            def ok():
                return True

            @staticmethod
            def spin_once(_node, timeout_sec):
                return None

            @staticmethod
            def shutdown():
                return None

        node = MagicMock()
        node.current_pose.side_effect = FakeTransformException()
        node.get_clock.side_effect = AssertionError(
            "simulated ROS time must not bound DDS discovery"
        )

        with patch.object(exploration_planner, "rclpy", FakeRclpy), patch.object(
            exploration_planner,
            "CurrentTfPoseReader",
            return_value=node,
        ), patch.object(
            exploration_planner,
            "TransformException",
            FakeTransformException,
        ), patch.object(
            exploration_planner.time,
            "monotonic",
            side_effect=(10.0, 11.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "timed out waiting for TF"):
                exploration_planner.read_current_tf_pose(
                    target_frame="odom",
                    source_frame="base_footprint",
                    timeout_sec=0.5,
                    lookup_timeout_sec=0.2,
                    use_sim_time=True,
                )

        node.get_clock.assert_not_called()
        node.destroy_node.assert_called_once_with()

    def test_start_pose_from_explicit_args(self):
        args = build_detected_stand_exploration_parser().parse_args(
            [
                "--map",
                "map.yaml",
                "--start-x",
                "-0.461",
                "--start-y",
                "0.365",
                "--start-yaw",
                "-0.270",
            ]
        )

        pose = start_pose_from_args(args)

        self.assertAlmostEqual(pose.x_m, -0.461)
        self.assertAlmostEqual(pose.y_m, 0.365)
        self.assertAlmostEqual(pose.yaw_rad, -0.270)

    def test_start_pose_requires_coordinates_without_tf(self):
        args = build_detected_stand_exploration_parser().parse_args(["--map", "map.yaml"])

        with self.assertRaisesRegex(ValueError, "start-x"):
            start_pose_from_args(args)


class DetectedStationExplorationTest(unittest.TestCase):
    def test_observation_snapshot_hashes_the_exact_parsed_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.jsonl"
            write_observation_jsonl(path, (observation(1),))
            expected_bytes = path.read_bytes()

            loaded, digest = load_observation_jsonl_snapshot(path)

        self.assertEqual(loaded, (observation(1),))
        self.assertEqual(digest, hashlib.sha256(expected_bytes).hexdigest())

    def test_observation_jsonl_rejects_ambiguous_schema_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observations.jsonl"
            payload = observation_to_payload(observation(1))
            payload["provenance"]["schema_version"] = 2.5
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "schema_version must be an integer"):
                load_observation_jsonl_snapshot(path)

            path.write_text('{"observation_id":"a","observation_id":"b"}\n')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_observation_jsonl_snapshot(path)

    def test_candidate_transforms_from_scan_frame_to_map_frame(self):
        obs = observation_from_candidate(
            candidate(),
            transform_scan_to_map=PlanarTransform(1.0, 2.0, math.pi / 2.0),
            observed_at_sec=12.0,
            provenance=provenance(),
            observation_index=1,
        )

        self.assertAlmostEqual(obs.x_m, 1.0, places=6)
        self.assertAlmostEqual(obs.y_m, 3.0, places=6)
        self.assertEqual(obs.provenance.scan_frame, "base_scan")

    def test_accumulator_confirms_after_min_hits_and_selects_first_confirmed(self):
        accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(
                merge_distance_m=0.2,
                min_hits=3,
                max_age_sec=5.0,
                min_confidence=0.5,
            )
        )
        first = [
            observation(1, x=0.50, y=0.50, observed_at=1.0),
            observation(2, x=0.52, y=0.49, observed_at=2.0),
            observation(3, x=0.51, y=0.51, observed_at=3.0),
        ]
        second = [
            observation(4, x=-0.50, y=0.20, observed_at=1.5),
            observation(5, x=-0.49, y=0.21, observed_at=2.5),
            observation(6, x=-0.48, y=0.22, observed_at=3.5),
        ]

        confirmed = accumulator.add_observations(first + second)
        selected = select_first_confirmed_stand(confirmed)

        self.assertEqual(len(confirmed), 2)
        self.assertAlmostEqual(selected.first_confirmed_at_sec, 3.0)
        self.assertEqual(selected.hit_count, 3)

        with self.assertRaisesRegex(ValueError, "ambiguous confirmed stands"):
            select_unique_confirmed_stand(confirmed)
        self.assertEqual(
            select_confirmed_stand_by_id(confirmed, "detected_stand_02").stand_id,
            "detected_stand_02",
        )

    def test_accumulator_rejects_low_confidence_and_expired_tracks(self):
        accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(
                merge_distance_m=0.2,
                min_hits=2,
                max_age_sec=1.0,
                min_confidence=0.7,
            )
        )
        low = observation(1, observed_at=1.0)
        low = low.__class__(**{**low.__dict__, "confidence": 0.2})
        accumulator.add_observation(low)
        accumulator.add_observation(observation(2, observed_at=10.0))
        confirmed = accumulator.add_observation(observation(3, observed_at=12.0))

        self.assertEqual(confirmed, ())

    def test_accumulator_rejects_repeated_wall_returns_by_boundary_clearance(self):
        accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(min_hits=3, min_confidence=0.5)
        )
        wall_returns = [
            observation(index, x=-1.93 + offset, y=-0.40, observed_at=float(index))
            for index, offset in enumerate((0.0, 0.01, -0.005), start=1)
        ]

        confirmed = accumulator.add_observations(wall_returns)

        self.assertEqual(confirmed, ())
        self.assertFalse(accumulator.accepts_observation(wall_returns[0]))

    def test_accumulator_keeps_legitimate_stand_clear_of_boundary(self):
        accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(min_hits=3, min_confidence=0.5)
        )
        stand_returns = [
            observation(index, x=0.405 + offset, y=0.685, observed_at=float(index))
            for index, offset in enumerate((0.0, 0.01, -0.005), start=1)
        ]

        confirmed = accumulator.add_observations(stand_returns)

        self.assertEqual(len(confirmed), 1)
        self.assertTrue(accumulator.accepts_observation(stand_returns[0]))

    def test_confirmed_stand_converts_to_station_layout(self):
        accumulator = StandConfirmationAccumulator(
            config=StandConfirmationConfig(min_hits=1, min_confidence=0.5)
        )
        stand = accumulator.add_observation(observation(1))[0]

        station = station_from_confirmed_stand(
            stand,
            config=DetectedStationLayoutConfig(
                station_id="A",
                approach_offset_m=0.3,
                keepout_radius_m=0.2,
                stand_yaw_rad=0.0,
            ),
        )

        self.assertEqual(station.station_id, "A")
        self.assertAlmostEqual(station.pose.x_m, stand.x_m)

    def test_provenance_validation_rejects_stale_or_wrong_frame_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_free_map(Path(tmpdir))
            stale = observation(1, map_yaml=map_yaml)
            stale = stale.__class__(
                **{
                    **stale.__dict__,
                    "provenance": provenance(map_yaml=map_yaml, tf_age_sec=5.0),
                }
            )
            with self.assertRaisesRegex(ValueError, "TF timestamp is stale"):
                validate_observation_provenance(
                    stale,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                )

            wrong_frame = observation(2, map_yaml=map_yaml)
            wrong_frame = wrong_frame.__class__(
                **{
                    **wrong_frame.__dict__,
                    "provenance": provenance(map_yaml=map_yaml, map_frame="odom"),
                }
            )
            with self.assertRaisesRegex(ValueError, "map_frame"):
                validate_observation_provenance(
                    wrong_frame,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                )

    def test_schema_v1_remains_loadable_but_is_not_planning_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_free_map(Path(tmpdir))
            payload = observation_to_payload(observation(1, map_yaml=map_yaml))
            provenance_payload = payload["provenance"]
            provenance_payload["schema_version"] = 1
            for key in (
                "observer_clock",
                "observer_clock_sec",
                "scan_age_sec",
                "tf_scan_skew_sec",
                "tf_query_stamp_sec",
                "tf_lookup_mode",
            ):
                provenance_payload.pop(key)

            loaded = observation_from_payload(payload)

            self.assertEqual(loaded.provenance.schema_version, 1)
            with self.assertRaisesRegex(ValueError, "unsupported observation schema"):
                validate_observation_provenance(
                    loaded,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                )

    def test_provenance_recomputes_instead_of_trusting_stored_ages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_free_map(Path(tmpdir))
            obs = observation(1, map_yaml=map_yaml)
            tampered = replace(
                obs,
                provenance=replace(obs.provenance, scan_age_sec=0.0),
            )

            with self.assertRaisesRegex(ValueError, "inconsistent scan_age_sec"):
                validate_observation_provenance(
                    tampered,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                )

    def test_provenance_requires_clock_binding_and_exact_query_stamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_free_map(Path(tmpdir))
            obs = observation(1, map_yaml=map_yaml)
            wrong_clock = replace(
                obs,
                provenance=replace(
                    obs.provenance,
                    observer_clock="ros_sim_time",
                ),
            )
            wrong_query = replace(
                obs,
                provenance=replace(
                    obs.provenance,
                    tf_query_stamp_sec=obs.provenance.scan_stamp_sec + 0.01,
                ),
            )

            for invalid, expected in (
                (wrong_clock, "clock/use_sim_time mismatch"),
                (wrong_query, "TF query/scan timestamp mismatch"),
            ):
                with self.subTest(expected=expected):
                    with self.assertRaisesRegex(ValueError, expected):
                        validate_observation_provenance(
                            invalid,
                            map_yaml=map_yaml,
                            required_map_frame="map",
                            required_base_frame="base_footprint",
                            required_localization_source="amcl",
                            max_tf_age_sec=1.0,
                        )

    def test_stream_rejects_mixed_observer_clock_domains(self):
        first = observation(1)
        second = observation(2)
        second = replace(
            second,
            provenance=replace(
                second.provenance,
                observer_clock="ros_sim_time",
                runtime_config={
                    **second.provenance.runtime_config,
                    "use_sim_time": True,
                },
            ),
        )

        with self.assertRaisesRegex(ValueError, "mixes incompatible observer clocks"):
            validated_observation_stream_clock((first, second))

    def test_all_observation_consumers_expose_the_same_timing_policy_cli(self):
        parsers_and_required_args = (
            (
                build_first_detected_station_parser(),
                [
                    "--map",
                    "map.yaml",
                    "--start-x",
                    "0",
                    "--start-y",
                    "0",
                    "--confirmation-json",
                    "confirmation.json",
                ],
            ),
            (
                build_detected_station_confirmation_parser(),
                [
                    "--map",
                    "map.yaml",
                    "--station-id",
                    "A",
                    "--confirmation-source",
                    "operator",
                ],
            ),
            (build_detected_stand_exploration_parser(), ["--map", "map.yaml"]),
        )
        timing_args = [
            "--max-scan-age-sec",
            "0.6",
            "--max-tf-age-sec",
            "0.7",
            "--max-future-timestamp-sec",
            "0.08",
            "--max-tf-scan-skew-sec",
            "0.009",
            "--required-observer-clock",
            "ros_sim_time",
        ]

        for parser, required in parsers_and_required_args:
            with self.subTest(prog=parser.prog):
                args = parser.parse_args(required + timing_args)
                self.assertEqual(args.max_scan_age_sec, 0.6)
                self.assertEqual(args.max_tf_age_sec, 0.7)
                self.assertEqual(args.max_future_timestamp_sec, 0.08)
                self.assertEqual(args.max_tf_scan_skew_sec, 0.009)
                self.assertEqual(args.required_observer_clock, "ros_sim_time")

    def test_provenance_validation_requires_map_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_free_map(Path(tmpdir))
            obs = observation(1)

            with self.assertRaisesRegex(ValueError, "map hash"):
                validate_observation_provenance(
                    obs,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                )

    def test_provenance_validation_can_bind_the_already_frozen_map_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            map_yaml = write_free_map(Path(tmpdir))
            obs = observation(1, map_yaml=map_yaml)
            frozen_yaml_sha256 = obs.provenance.map_yaml_sha256
            map_yaml.write_text(map_yaml.read_text() + "# later revision\n")

            validate_observation_provenance(
                obs,
                map_yaml=map_yaml,
                required_map_frame="map",
                required_base_frame="base_footprint",
                required_localization_source="amcl",
                max_tf_age_sec=1.0,
                expected_map_yaml_sha256=frozen_yaml_sha256,
            )

            with self.assertRaisesRegex(ValueError, "map hash"):
                validate_observation_provenance(
                    obs,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                )

    def test_provenance_rejects_changed_map_image_with_same_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            obs = observation(1, map_yaml=map_yaml)
            old_yaml_hash = obs.provenance.map_yaml_sha256
            (root / "map.pgm").write_text(
                "P2\n30 30\n255\n" + " ".join(["0"] * 900) + "\n"
            )
            current = freeze_map_bundle(
                map_yaml,
                semantic_map_id=map_yaml.stem,
                planning_frame="map",
            )

            self.assertEqual(current.yaml_sha256, old_yaml_hash)
            with self.assertRaisesRegex(ValueError, "map bundle hash"):
                validate_observation_provenance(
                    obs,
                    map_yaml=map_yaml,
                    required_map_frame="map",
                    required_base_frame="base_footprint",
                    required_localization_source="amcl",
                    max_tf_age_sec=1.0,
                    expected_map_yaml_sha256=current.yaml_sha256,
                    expected_map_bundle_sha256=current.bundle_sha256,
                )

    def test_confirmation_receipt_binds_selected_stand_to_station(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            receipt = root / "confirmation.json"
            write_confirmation(receipt)
            accumulator = StandConfirmationAccumulator(
                config=StandConfirmationConfig(min_hits=1, min_confidence=0.5)
            )
            stand = accumulator.add_observation(observation(1))[0]

            payload = load_and_validate_confirmation_receipt(receipt, stand=stand, station_id="A")

            self.assertEqual(payload["station_id"], "A")

            write_confirmation(receipt, station_id="B")
            with self.assertRaisesRegex(ValueError, "station_id mismatch"):
                load_and_validate_confirmation_receipt(receipt, stand=stand, station_id="A")

    def test_create_detected_station_confirmation_writes_operator_receipt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                ],
            )
            confirmation_json = root / "confirmation.json"

            status = create_detected_station_confirmation_main(
                [
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--station-id",
                    "A",
                    "--confirmation-source",
                    "operator",
                    "--operator-confirmed",
                    "--output-json",
                    str(confirmation_json),
                ]
            )

            self.assertEqual(status, 0)
            payload = json.loads(confirmation_json.read_text())
            self.assertEqual(payload["confirmation_source"], "operator")
            self.assertTrue(payload["operator_confirmed"])
            self.assertEqual(payload["station_id"], "A")
            self.assertEqual(payload["stand_id"], "detected_stand_01")

    def test_create_detected_station_confirmation_rejects_unconfirmed_operator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                ],
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_detected_station_confirmation_main(
                        [
                            "--observations-jsonl",
                            str(observations_jsonl),
                            "--map",
                            str(map_yaml),
                            "--station-id",
                            "A",
                            "--confirmation-source",
                            "operator",
                            "--output-json",
                            str(root / "confirmation.json"),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)

    def test_create_detected_station_confirmation_rejects_ambiguous_stands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                    observation(4, x=-0.5, y=0.2, observed_at=1.0, map_yaml=map_yaml),
                    observation(5, x=-0.49, y=0.21, observed_at=2.0, map_yaml=map_yaml),
                    observation(6, x=-0.48, y=0.2, observed_at=3.0, map_yaml=map_yaml),
                ],
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    create_detected_station_confirmation_main(
                        [
                            "--observations-jsonl",
                            str(observations_jsonl),
                            "--map",
                            str(map_yaml),
                            "--station-id",
                            "A",
                            "--confirmation-source",
                            "operator",
                            "--operator-confirmed",
                            "--output-json",
                            str(root / "confirmation.json"),
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)

    def test_create_detected_station_confirmation_accepts_explicit_stand_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                    observation(4, x=-0.5, y=0.2, observed_at=1.0, map_yaml=map_yaml),
                    observation(5, x=-0.49, y=0.21, observed_at=2.0, map_yaml=map_yaml),
                    observation(6, x=-0.48, y=0.2, observed_at=3.0, map_yaml=map_yaml),
                ],
            )
            confirmation_json = root / "confirmation.json"

            status = create_detected_station_confirmation_main(
                [
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--station-id",
                    "A",
                    "--stand-id",
                    "detected_stand_02",
                    "--confirmation-source",
                    "operator",
                    "--operator-confirmed",
                    "--output-json",
                    str(confirmation_json),
                ]
            )

            self.assertEqual(status, 0)
            payload = json.loads(confirmation_json.read_text())
            self.assertEqual(payload["stand_id"], "detected_stand_02")

    def test_plan_first_detected_station_writes_layout_route_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                ],
            )
            layout_json = root / "layout.json"
            route_csv = root / "route.csv"
            diagnostics_json = root / "diagnostics.json"
            confirmation_json = root / "confirmation.json"
            write_confirmation(confirmation_json)

            status = plan_first_detected_station_main(
                [
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--start-x",
                    "0.0",
                    "--start-y",
                    "0.0",
                    "--confirmation-json",
                    str(confirmation_json),
                    "--layout-json",
                    str(layout_json),
                    "--layout-csv",
                    str(root / "layout.csv"),
                    "--route-csv",
                    str(route_csv),
                    "--diagnostics-json",
                    str(diagnostics_json),
                ]
            )

            self.assertEqual(status, 0)
            self.assertTrue(route_csv.exists())
            layout = json.loads(layout_json.read_text())
            diagnostics = json.loads(diagnostics_json.read_text())
            self.assertEqual(layout["stations"][0]["station_id"], "A")
            self.assertEqual(
                diagnostics["metadata"]["detected_station"]["stand_hit_count"],
                3,
            )
            self.assertEqual(
                diagnostics["metadata"]["detected_station"]["confirmation"]["station_id"],
                "A",
            )
            pre_approach = diagnostics["metadata"]["detected_station"]["pre_approach"]
            self.assertFalse(pre_approach["hidden_stand_yaw_used"])
            self.assertEqual(
                pre_approach["orientation_source"],
                "robot_to_detected_stand_bearing",
            )
            with route_csv.open(newline="") as route_file:
                route_rows = list(csv.DictReader(route_file))
            self.assertAlmostEqual(
                float(route_rows[-1]["yaw_rad"]),
                float(pre_approach["yaw_rad"]),
            )

    def test_plan_first_detected_station_rejects_missing_confirmation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                ],
            )

            with redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    plan_first_detected_station_main(
                        [
                            "--observations-jsonl",
                            str(observations_jsonl),
                            "--map",
                            str(map_yaml),
                            "--start-x",
                            "0.0",
                            "--start-y",
                            "0.0",
                        ]
                    )

            self.assertEqual(raised.exception.code, 2)

    def test_plan_detected_stand_exploration_writes_multi_leg_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                    observation(4, x=-0.2, y=0.4, observed_at=1.1, map_yaml=map_yaml),
                    observation(5, x=-0.21, y=0.4, observed_at=2.1, map_yaml=map_yaml),
                    observation(6, x=-0.19, y=0.4, observed_at=3.1, map_yaml=map_yaml),
                ],
            )
            route_csv = root / "explore_route.csv"
            diagnostics_json = root / "explore_diagnostics.json"

            status = plan_detected_stand_exploration_main(
                [
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--start-x",
                    "0.0",
                    "--start-y",
                    "0.0",
                    "--exploration-state-json",
                    str(root / "explore_state.json"),
                    "--layout-json",
                    str(root / "explore_layout.json"),
                    "--layout-csv",
                    str(root / "explore_layout.csv"),
                    "--route-csv",
                    str(route_csv),
                    "--diagnostics-json",
                    str(diagnostics_json),
                ]
            )

            self.assertEqual(status, 0)
            diagnostics = json.loads(diagnostics_json.read_text())
            self.assertEqual(len(diagnostics["legs"]), 1)
            self.assertEqual(diagnostics["metadata"]["plan_mode"], "next-candidate")
            self.assertEqual(diagnostics["metadata"]["stand_count"], 1)
            self.assertEqual(diagnostics["metadata"]["pending_candidate_count"], 2)
            self.assertIn("selected_candidate_stand_id", diagnostics["metadata"])
            self.assertIn("leg_index", route_csv.read_text())

    def test_plan_detected_stand_exploration_can_still_write_full_route_explicitly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                    observation(4, x=-0.2, y=0.4, observed_at=1.1, map_yaml=map_yaml),
                    observation(5, x=-0.21, y=0.4, observed_at=2.1, map_yaml=map_yaml),
                    observation(6, x=-0.19, y=0.4, observed_at=3.1, map_yaml=map_yaml),
                ],
            )
            route_csv = root / "explore_route.csv"
            diagnostics_json = root / "explore_diagnostics.json"

            status = plan_detected_stand_exploration_main(
                [
                    "--plan-mode",
                    "full-route",
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--start-x",
                    "0.0",
                    "--start-y",
                    "0.0",
                    "--exploration-state-json",
                    str(root / "explore_state.json"),
                    "--layout-json",
                    str(root / "explore_layout.json"),
                    "--layout-csv",
                    str(root / "explore_layout.csv"),
                    "--route-csv",
                    str(route_csv),
                    "--diagnostics-json",
                    str(diagnostics_json),
                ]
            )

            self.assertEqual(status, 0)
            diagnostics = json.loads(diagnostics_json.read_text())
            self.assertEqual(len(diagnostics["legs"]), 2)
            self.assertEqual(diagnostics["metadata"]["plan_mode"], "full-route")
            self.assertEqual(diagnostics["metadata"]["stand_count"], 2)

    def test_plan_detected_stand_exploration_skips_rejected_candidate_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                    observation(4, x=-0.2, y=0.4, observed_at=1.1, map_yaml=map_yaml),
                    observation(5, x=-0.21, y=0.4, observed_at=2.1, map_yaml=map_yaml),
                    observation(6, x=-0.19, y=0.4, observed_at=3.1, map_yaml=map_yaml),
                ],
            )
            state_json = root / "explore_state.json"
            diagnostics_json = root / "explore_diagnostics.json"

            status = plan_detected_stand_exploration_main(
                [
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--start-x",
                    "0.0",
                    "--start-y",
                    "0.0",
                    "--exploration-state-json",
                    str(state_json),
                    "--mark-rejected-stand-id",
                    "detected_stand_01",
                    "--layout-json",
                    str(root / "explore_layout.json"),
                    "--layout-csv",
                    str(root / "explore_layout.csv"),
                    "--route-csv",
                    str(root / "explore_route.csv"),
                    "--diagnostics-json",
                    str(diagnostics_json),
                ]
            )

            self.assertEqual(status, 0)
            diagnostics = json.loads(diagnostics_json.read_text())
            state = json.loads(state_json.read_text())
            self.assertEqual(diagnostics["metadata"]["selected_candidate_stand_id"], "detected_stand_02")
            self.assertEqual(state["decisions"][0]["status"], "rejected")
            self.assertEqual(state["decisions"][0]["stand_id"], "detected_stand_01")

    def test_plan_detected_stand_exploration_reconciles_state_by_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            map_yaml = write_free_map(root)
            observations_jsonl = root / "observations.jsonl"
            state_json = root / "explore_state.json"
            state_json.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "decisions": [
                            {
                                "status": "rejected",
                                "stand_id": "previous_id",
                                "x_m": 0.5,
                                "y_m": 0.5,
                                "confidence": 0.8,
                                "hit_count": 3,
                                "source_observation_ids": [],
                            }
                        ],
                        "confirmed_stand_ids": [],
                        "rejected_stand_ids": [],
                    }
                )
            )
            write_observation_jsonl(
                observations_jsonl,
                [
                    observation(1, x=0.5, y=0.5, observed_at=1.0, map_yaml=map_yaml),
                    observation(2, x=0.51, y=0.5, observed_at=2.0, map_yaml=map_yaml),
                    observation(3, x=0.49, y=0.5, observed_at=3.0, map_yaml=map_yaml),
                    observation(4, x=-0.2, y=0.4, observed_at=1.1, map_yaml=map_yaml),
                    observation(5, x=-0.21, y=0.4, observed_at=2.1, map_yaml=map_yaml),
                    observation(6, x=-0.19, y=0.4, observed_at=3.1, map_yaml=map_yaml),
                ],
            )

            status = plan_detected_stand_exploration_main(
                [
                    "--observations-jsonl",
                    str(observations_jsonl),
                    "--map",
                    str(map_yaml),
                    "--start-x",
                    "0.0",
                    "--start-y",
                    "0.0",
                    "--exploration-state-json",
                    str(state_json),
                    "--layout-json",
                    str(root / "explore_layout.json"),
                    "--layout-csv",
                    str(root / "explore_layout.csv"),
                    "--route-csv",
                    str(root / "explore_route.csv"),
                    "--diagnostics-json",
                    str(root / "explore_diagnostics.json"),
                ]
            )

            self.assertEqual(status, 0)
            diagnostics = json.loads((root / "explore_diagnostics.json").read_text())
            self.assertEqual(diagnostics["metadata"]["selected_candidate_stand_id"], "detected_stand_02")
            self.assertEqual(diagnostics["metadata"]["rejected_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
