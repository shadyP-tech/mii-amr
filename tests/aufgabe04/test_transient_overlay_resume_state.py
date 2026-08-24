from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.aufgabe04.artifacts.content_store import payload_sha256
from scripts.aufgabe04.navigation.foundation.arena_bounds import ArenaBounds
from scripts.aufgabe04.navigation.planning.map_io import CELL_FREE, MapMetadata, OccupancyGrid
from scripts.aufgabe04.navigation.foundation.models import Pose2D
from scripts.aufgabe04.navigation.coverage.stand_blockage_replan import (
    TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
    TransientObstacleOverlay,
    write_transient_obstacle_overlay,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import (
    STATUS_PROVISIONAL,
    CoverageSurveyConfig,
    SurveyCandidate,
    build_coverage_survey_plan,
)
from scripts.aufgabe04.navigation.coverage.transient_overlay_resume_state import (
    TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_KEY,
    TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD,
    add_adopted_route_hash,
    bind_transient_overlay_resume_state_to_diagnostics,
    load_jsonl_event_objects,
    load_transient_overlay_resume_state,
    refresh_transient_overlay_resume_state,
    transient_overlay_resume_state_sha256,
    update_transient_overlay_resume_state_from_events,
    validate_transient_overlay_resume_state_diagnostics_binding,
    write_transient_overlay_resume_state,
)


def _free_grid() -> OccupancyGrid:
    width = 50
    height = 30
    return OccupancyGrid(
        metadata=MapMetadata(
            yaml_path=Path("map.yaml"),
            image_path=Path("map.pgm"),
            resolution=0.10,
            origin=(-2.5, -1.5, 0.0),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.20,
            mode="trinary",
        ),
        width=width,
        height=height,
        cells=tuple(tuple([CELL_FREE] * width) for _ in range(height)),
    )


def _plan():
    return build_coverage_survey_plan(
        _free_grid(),
        map_bundle_sha256="a" * 64,
        start=Pose2D(-1.5, 0.0, 0.0),
        survey_id="resume_state_test",
        arena_bounds=ArenaBounds(length_m=4.0, width_m=2.0),
        config=CoverageSurveyConfig(lane_count=1, stop_spacing_m=0.70),
    )


def _candidate(index: int) -> SurveyCandidate:
    return SurveyCandidate(
        candidate_uid=f"transient_obstacle_{index:04d}",
        x_m=-0.7 + 0.2 * index,
        y_m=0.05 * index,
        radius_m=0.06,
        uncertainty_m=0.02,
        keepout_radius_m=0.34,
        confidence=1.0,
        hit_count=1,
        first_seen_sec=0.0,
        last_seen_sec=0.0,
        source_observation_ids=(f"blockage_{index:04d}",),
        viewpoint_ids=(),
        status=STATUS_PROVISIONAL,
    )


class TransientOverlayResumeStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = _plan()
        self.target = self.plan.viewpoints[0].viewpoint_id
        self.overlay_1 = self._write_overlay("overlay_1.json", (1,))
        self.route_1 = self._write_route("route_1.csv", "route one\n")

    def _write_overlay(self, name: str, indices: tuple[int, ...]) -> Path:
        path = self.root / name
        overlay = TransientObstacleOverlay(
            schema_version=TRANSIENT_OBSTACLE_OVERLAY_SCHEMA_VERSION,
            survey_id=self.plan.survey_id,
            planning_frame=self.plan.planning_frame,
            map_bundle_sha256=self.plan.map_bundle_sha256,
            candidates=tuple(_candidate(index) for index in indices),
        )
        write_transient_obstacle_overlay(
            path,
            overlay,
            source={"kind": "test"},
        )
        return path

    def _write_route(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def _event(
        self,
        *,
        index: int = 1,
        overlay: Path | None = None,
        route: Path | None = None,
        target: str | None = None,
        run_id: str = "child_001",
        semantic: bool = False,
    ) -> dict[str, object]:
        selected_overlay = overlay or self.overlay_1
        selected_route = route or self.route_1
        return {
            "event": "transient_navigation_blockage_replanned",
            "run_id": run_id,
            "leg_index": 0,
            "replan_index": index,
            "target_viewpoint_id": target or self.target,
            "semantic_survey_evidence": semantic,
            "transient_obstacle_overlay_json": str(
                selected_overlay.relative_to(self.root)
            ),
            "replacement_route_csv": str(selected_route.relative_to(self.root)),
            "source_map_route_sha256": hashlib.sha256(
                selected_route.read_bytes()
            ).hexdigest(),
        }

    def _first_state(self):
        state = update_transient_overlay_resume_state_from_events(
            [self._event()],
            plan=self.plan,
            coverage_leg_index=0,
            target_viewpoint_id=self.target,
            max_replans=3,
            artifact_root=self.root,
        )
        self.assertIsNotNone(state)
        return state

    def test_valid_first_state_is_canonical_immutable_and_round_trips(self):
        state = self._first_state()

        self.assertEqual(state.completed_replan_count, 1)
        self.assertEqual(state.remaining_replans, 2)
        self.assertEqual(state.overlay_candidate_ids, ("transient_obstacle_0001",))
        self.assertEqual(state.transient_obstacle_overlay_path, str(self.overlay_1))
        self.assertEqual(state.adopted_route_paths, (str(self.route_1),))
        self.assertEqual(state.source_run_ids, ("child_001",))
        self.assertFalse(state.semantic_survey_evidence)
        self.assertFalse(state.motion_continues_authorized)
        self.assertFalse(state.automatic_motion_authorized)

        state_path = self.root / "resume_state.json"
        digest = write_transient_overlay_resume_state(
            state_path, state, plan=self.plan
        )
        self.assertEqual(digest, transient_overlay_resume_state_sha256(state))
        loaded = load_transient_overlay_resume_state(
            state_path,
            plan=self.plan,
            expected_coverage_leg_index=0,
            expected_target_viewpoint_id=self.target,
            expected_max_replans=3,
        )
        self.assertEqual(loaded, state)

    def test_previous_state_with_no_new_events_is_returned_unchanged(self):
        previous = self._first_state()

        current = update_transient_overlay_resume_state_from_events(
            [],
            plan=self.plan,
            coverage_leg_index=0,
            target_viewpoint_id=self.target,
            max_replans=3,
            artifact_root=self.root,
            previous_state=previous,
        )

        self.assertIs(current, previous)

    def test_cumulative_event_extends_overlay_route_hashes_and_budget(self):
        previous = self._first_state()
        overlay_2 = self._write_overlay("overlay_2.json", (1, 2))
        route_2 = self._write_route("route_2.csv", "route two\n")

        current = update_transient_overlay_resume_state_from_events(
            [
                self._event(
                    index=2,
                    overlay=overlay_2,
                    route=route_2,
                    run_id="child_002",
                )
            ],
            plan=self.plan,
            coverage_leg_index=0,
            target_viewpoint_id=self.target,
            max_replans=3,
            artifact_root=self.root,
            previous_state=previous,
        )

        self.assertEqual(current.completed_replan_count, 2)
        self.assertEqual(current.remaining_replans, 1)
        self.assertEqual(
            current.overlay_candidate_ids,
            ("transient_obstacle_0001", "transient_obstacle_0002"),
        )
        self.assertEqual(current.source_run_ids, ("child_001", "child_002"))
        self.assertEqual(len(set(current.adopted_route_sha256s)), 2)

    def test_refresh_and_add_helpers_preserve_cumulative_contract(self):
        previous = self._first_state()
        overlay_2 = self._write_overlay("overlay_2.json", (1, 2))
        route_2 = self._write_route("route_2.csv", "route two\n")

        refreshed = refresh_transient_overlay_resume_state(
            previous,
            overlay_path=overlay_2,
            plan=self.plan,
            artifact_root=self.root,
        )
        updated = add_adopted_route_hash(
            refreshed,
            route_path=route_2,
            source_run_id="child_002",
            replan_index=2,
            plan=self.plan,
            artifact_root=self.root,
        )

        self.assertEqual(updated.completed_replan_count, 2)
        self.assertEqual(updated.transient_obstacle_overlay_path, str(overlay_2))
        self.assertEqual(updated.adopted_route_paths[-1], str(route_2))

    def test_rejects_gap_wrong_target_and_semantic_evidence(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            update_transient_overlay_resume_state_from_events(
                [self._event(index=2)],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )
        with self.assertRaisesRegex(ValueError, "another target"):
            update_transient_overlay_resume_state_from_events(
                [self._event(target=self.plan.viewpoints[1].viewpoint_id)],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )
        with self.assertRaisesRegex(ValueError, "semantic survey evidence"):
            update_transient_overlay_resume_state_from_events(
                [self._event(semantic=True)],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

        malformed = self._event()
        del malformed["replan_index"]
        with self.assertRaisesRegex(ValueError, "replan_index must be an integer"):
            update_transient_overlay_resume_state_from_events(
                [malformed],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

    def test_rejects_nonmonotonic_overlay_and_declared_route_hash_mismatch(self):
        previous = self._first_state()
        nonmonotonic = self._write_overlay("nonmonotonic.json", (2,))
        route_2 = self._write_route("route_2.csv", "route two\n")
        with self.assertRaisesRegex(ValueError, "monotonic extension"):
            update_transient_overlay_resume_state_from_events(
                [self._event(index=2, overlay=nonmonotonic, route=route_2)],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
                previous_state=previous,
            )

        event = self._event()
        event["source_map_route_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "route hash mismatch"):
            update_transient_overlay_resume_state_from_events(
                [event],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

        missing_hash = self._event()
        del missing_hash["source_map_route_sha256"]
        with self.assertRaisesRegex(ValueError, "source_map_route_sha256"):
            update_transient_overlay_resume_state_from_events(
                [missing_hash],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

    def test_rejects_missing_overlay_or_route(self):
        missing_overlay = dict(self._event())
        missing_overlay["transient_obstacle_overlay_json"] = "missing.json"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            update_transient_overlay_resume_state_from_events(
                [missing_overlay],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

        missing_route = dict(self._event())
        missing_route["replacement_route_csv"] = "missing.csv"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            update_transient_overlay_resume_state_from_events(
                [missing_route],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

    def test_load_rejects_tampered_overlay_and_route(self):
        state = self._first_state()
        overlay_state_path = self.root / "overlay_state.json"
        write_transient_overlay_resume_state(
            overlay_state_path, state, plan=self.plan
        )
        self.overlay_1.write_text(
            self.overlay_1.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "overlay hash mismatch"):
            load_transient_overlay_resume_state(
                overlay_state_path, plan=self.plan
            )

        # Build an independent state because the first overlay is now altered.
        overlay_2 = self._write_overlay("fresh_overlay.json", (1,))
        route_2 = self._write_route("fresh_route.csv", "fresh route\n")
        route_state = update_transient_overlay_resume_state_from_events(
            [self._event(overlay=overlay_2, route=route_2)],
            plan=self.plan,
            coverage_leg_index=0,
            target_viewpoint_id=self.target,
            max_replans=3,
            artifact_root=self.root,
        )
        route_state_path = self.root / "route_state.json"
        write_transient_overlay_resume_state(
            route_state_path, route_state, plan=self.plan
        )
        route_2.write_text("tampered route\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "adopted route 1 hash mismatch"):
            load_transient_overlay_resume_state(route_state_path, plan=self.plan)

    def test_rejects_symlinks_and_noncanonical_event_paths(self):
        overlay_link = self.root / "overlay_link.json"
        os.symlink(self.overlay_1.name, overlay_link)
        event = self._event()
        event["transient_obstacle_overlay_json"] = overlay_link.name
        with self.assertRaisesRegex(ValueError, "symlink"):
            update_transient_overlay_resume_state_from_events(
                [event],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

        event = self._event()
        event["replacement_route_csv"] = "nested/../route_1.csv"
        with self.assertRaisesRegex(ValueError, "not canonical"):
            update_transient_overlay_resume_state_from_events(
                [event],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
            )

    def test_load_rejects_hash_tamper_unknown_fields_and_noncanonical_paths(self):
        state = self._first_state()
        state_path = self.root / "state.json"
        write_transient_overlay_resume_state(state_path, state, plan=self.plan)

        tampered = json.loads(state_path.read_text(encoding="utf-8"))
        tampered["remaining_replans"] = 1
        state_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_transient_overlay_resume_state(state_path, plan=self.plan)

        unknown_path = self.root / "unknown.json"
        unknown = json.loads(
            (self.root / "state.json").read_text(encoding="utf-8")
        )
        unknown["remaining_replans"] = 2
        unknown["unknown"] = True
        unknown.pop(TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD)
        unknown[TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD] = payload_sha256(unknown)
        unknown_path.write_text(json.dumps(unknown), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, r"unknown=\['unknown'\]"):
            load_transient_overlay_resume_state(unknown_path, plan=self.plan)

        noncanonical_path = self.root / "noncanonical.json"
        noncanonical = json.loads(unknown_path.read_text(encoding="utf-8"))
        noncanonical.pop("unknown")
        noncanonical["transient_obstacle_overlay_path"] = str(
            self.root / "nested" / ".." / self.overlay_1.name
        )
        noncanonical.pop(TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD)
        noncanonical[TRANSIENT_OVERLAY_RESUME_STATE_HASH_FIELD] = payload_sha256(
            noncanonical
        )
        noncanonical_path.write_text(json.dumps(noncanonical), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "canonical absolute"):
            load_transient_overlay_resume_state(noncanonical_path, plan=self.plan)

    def test_immutable_write_is_idempotent_but_rejects_different_state(self):
        state = self._first_state()
        state_path = self.root / "state.json"
        first = write_transient_overlay_resume_state(
            state_path, state, plan=self.plan
        )
        second = write_transient_overlay_resume_state(
            state_path, state, plan=self.plan
        )
        self.assertEqual(first, second)

        different = replace(state, max_replans=4, remaining_replans=3)
        with self.assertRaisesRegex(ValueError, "refusing to replace immutable"):
            write_transient_overlay_resume_state(
                state_path, different, plan=self.plan
            )

    def test_jsonl_loader_and_run_filter(self):
        log = self.root / "events.jsonl"
        ignored = {"event": "heartbeat", "run_id": "other"}
        event = self._event()
        log.write_text(
            json.dumps(ignored) + "\n\n" + json.dumps(event) + "\n",
            encoding="utf-8",
        )
        loaded = load_jsonl_event_objects(log)
        self.assertEqual(loaded, (ignored, event))
        invocation_offset = len((json.dumps(ignored) + "\n").encode("utf-8"))
        self.assertEqual(
            load_jsonl_event_objects(log, start_offset=invocation_offset),
            (event,),
        )
        with self.assertRaisesRegex(ValueError, "complete-record boundary"):
            load_jsonl_event_objects(log, start_offset=1)
        state = update_transient_overlay_resume_state_from_events(
            log,
            plan=self.plan,
            coverage_leg_index=0,
            target_viewpoint_id=self.target,
            max_replans=3,
            artifact_root=self.root,
            source_run_id="child_001",
        )
        self.assertEqual(state.source_run_ids, ("child_001",))

        bad = self.root / "bad.jsonl"
        bad.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be an object"):
            load_jsonl_event_objects(bad)

        log_link = self.root / "events_link.jsonl"
        os.symlink(log.name, log_link)
        with self.assertRaisesRegex(ValueError, "normal file"):
            load_jsonl_event_objects(log_link)

    def test_event_artifacts_can_be_bound_to_exact_leg_replan_slots(self):
        survey_root = self.root / "survey"
        session_root = self.root / "session"
        overlay_dir = survey_root / "replans/leg_000_replan_001"
        route_dir = session_root / "execution/coverage_leg_000_replan_001"
        overlay_dir.mkdir(parents=True)
        route_dir.mkdir(parents=True)
        overlay = overlay_dir / "transient_obstacle_overlay.json"
        route = route_dir / "route.csv"
        shutil.copyfile(self.overlay_1, overlay)
        shutil.copyfile(self.route_1, route)
        event = self._event(overlay=overlay, route=route)
        event["transient_obstacle_overlay_json"] = str(overlay)
        event["replacement_route_csv"] = str(route)

        state = update_transient_overlay_resume_state_from_events(
            [event],
            plan=self.plan,
            coverage_leg_index=0,
            target_viewpoint_id=self.target,
            max_replans=3,
            artifact_root=self.root,
            expected_survey_root=survey_root,
            expected_session_root=session_root,
        )
        self.assertEqual(state.transient_obstacle_overlay_path, str(overlay))
        self.assertEqual(state.adopted_route_paths, (str(route),))

        wrong_route = self.root / "wrong_route.csv"
        shutil.copyfile(route, wrong_route)
        wrong = dict(event)
        wrong["replacement_route_csv"] = str(wrong_route)
        with self.assertRaisesRegex(ValueError, "leg/replan slot"):
            update_transient_overlay_resume_state_from_events(
                [wrong],
                plan=self.plan,
                coverage_leg_index=0,
                target_viewpoint_id=self.target,
                max_replans=3,
                artifact_root=self.root,
                expected_survey_root=survey_root,
                expected_session_root=session_root,
            )

    def test_diagnostics_binding_requires_motion_free_source_and_validates_exactly(self):
        state = self._first_state()
        state_path = self.root / "state.json"
        write_transient_overlay_resume_state(state_path, state, plan=self.plan)
        source = self.root / "source_diagnostics.json"
        source.write_text(
            json.dumps(
                {
                    "metadata": {
                        "motion_authorized": False,
                        "plan_sha256": state.coverage_plan_sha256,
                        "survey_id": state.survey_id,
                        "planning_frame": state.planning_frame,
                        "map_bundle_sha256": state.map_bundle_sha256,
                        "target_viewpoint_id": state.target_viewpoint_id,
                    },
                    "legs": [],
                }
            ),
            encoding="utf-8",
        )
        bound = self.root / "bound_diagnostics.json"
        digest = bind_transient_overlay_resume_state_to_diagnostics(
            source,
            bound,
            resume_state_path=state_path,
            plan=self.plan,
        )
        self.assertEqual(len(digest), 64)
        loaded = validate_transient_overlay_resume_state_diagnostics_binding(
            bound,
            resume_state_path=state_path,
            plan=self.plan,
            expected_coverage_leg_index=0,
            expected_target_viewpoint_id=self.target,
            expected_max_replans=3,
        )
        self.assertEqual(loaded, state)

        payload = json.loads(bound.read_text(encoding="utf-8"))
        payload["metadata"][TRANSIENT_OVERLAY_RESUME_DIAGNOSTICS_BINDING_KEY][
            "completed_replan_count"
        ] = 2
        tampered = self.root / "tampered_diagnostics.json"
        tampered.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "binding differs"):
            validate_transient_overlay_resume_state_diagnostics_binding(
                tampered,
                resume_state_path=state_path,
                plan=self.plan,
            )

        source_payload = json.loads(source.read_text(encoding="utf-8"))
        source_payload["metadata"]["motion_authorized"] = True
        motion_source = self.root / "motion_source.json"
        motion_source.write_text(json.dumps(source_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicitly motion-free"):
            bind_transient_overlay_resume_state_to_diagnostics(
                motion_source,
                self.root / "must_not_exist.json",
                resume_state_path=state_path,
                plan=self.plan,
            )


if __name__ == "__main__":
    unittest.main()
