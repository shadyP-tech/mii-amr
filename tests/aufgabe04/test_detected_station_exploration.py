from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.aufgabe04.navigation.plan_first_detected_station import (  # noqa: E402
    load_and_validate_confirmation_receipt,
    main as plan_first_detected_station_main,
    validate_observation_provenance,
)
from scripts.aufgabe04.navigation.plan_detected_stand_exploration import (  # noqa: E402
    build_parser as build_detected_stand_exploration_parser,
    main as plan_detected_stand_exploration_main,
    start_pose_from_args,
)
from scripts.aufgabe04.navigation.create_detected_station_confirmation import (  # noqa: E402
    main as create_detected_station_confirmation_main,
)
from scripts.aufgabe04.navigation.route_context import file_sha256  # noqa: E402
from scripts.aufgabe04.perception.models import StandCandidate  # noqa: E402
from scripts.aufgabe04.perception.stand_confirmation import (  # noqa: E402
    StandConfirmationAccumulator,
    StandConfirmationConfig,
    select_confirmed_stand_by_id,
    select_first_confirmed_stand,
    select_unique_confirmed_stand,
)
from scripts.aufgabe04.perception.stand_observation import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION,
    ObservationProvenance,
    PlanarTransform,
    observation_from_candidate,
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


def provenance(*, map_yaml: Path | None = None, tf_age_sec=0.1, map_frame="map"):
    return ObservationProvenance(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        observer_version="test-observer",
        resolved_scan_topic="/scan",
        scan_frame="base_scan",
        map_frame=map_frame,
        base_frame="base_footprint",
        localization_source="amcl",
        scan_stamp_sec=10.0,
        tf_lookup_stamp_sec=10.0,
        tf_age_sec=tf_age_sec,
        runtime_config={"scan_topic": "/scan"},
        map_yaml=str(map_yaml or ""),
        map_yaml_sha256=file_sha256(map_yaml) if map_yaml else "",
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
            with self.assertRaisesRegex(ValueError, "TF age"):
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
