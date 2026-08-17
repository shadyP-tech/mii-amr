from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.aufgabe04.navigation.stand_coverage_survey import (
    STATUS_PENDING_CAMERA,
)
from scripts.aufgabe04.real_robot import autonomous_coverage_mission as mission
from scripts.aufgabe04.real_robot.autonomous_coverage_mission import (
    CANDIDATE_SNAPSHOT_READY,
    RESUME_FROM_CHECKPOINT,
    CompletedCoverageLeg,
    CoverageCheckpointComplete,
    CoverageCheckpointIdentity,
    CoverageComplete,
    CoverageMissionConfig,
    CoverageMissionEffects,
    PublishedCoverageCheckpoint,
    execute_coverage_mission,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _identity(root: Path) -> CoverageCheckpointIdentity:
    return CoverageCheckpointIdentity(
        session_root=root / "session",
        session_id="coverage_mission_test",
        run_mode="execute-coverage-checkpoint",
        robot_id="turtlebot1",
        robot_profile_sha256=HASH_A,
        calibration_profile_sha256=HASH_B,
        physical_site_sha256=HASH_C,
        map_bundle_sha256=HASH_A,
        config_sha256=HASH_B,
    )


def _config(
    root: Path,
    *,
    leg_limit: int,
    expected_stand_count: int = 1,
    initial_leg_index: int = 0,
    parent_checkpoint: Path | None = None,
) -> CoverageMissionConfig:
    identity = _identity(root)
    identity.session_root.mkdir(parents=True)
    survey_root = identity.session_root / "survey"
    survey_root.mkdir()
    return CoverageMissionConfig(
        survey_root=survey_root,
        plan=SimpleNamespace(map_bundle_sha256=HASH_A),
        coverage_plan_path=survey_root / "coverage_plan.json",
        checkpoint_identity=identity,
        expected_stand_count=expected_stand_count,
        initial_leg_index=initial_leg_index,
        coverage_leg_limit=leg_limit,
        parent_checkpoint_path=parent_checkpoint,
    )


def _summary(
    next_viewpoint_id: str | None,
    *,
    visited: int,
    total: int = 3,
    ratio: float | None = None,
) -> dict[str, object]:
    return {
        "next_viewpoint_id": next_viewpoint_id,
        "recorded_viewpoint_id": (
            None if visited == 0 else f"survey_vp_{visited:03d}"
        ),
        "coverage_complete": next_viewpoint_id is None,
        "visited_coverage_ratio": (
            visited / total if ratio is None else ratio
        ),
        "visited_viewpoint_count": visited,
        "total_viewpoint_count": total,
        "candidate_counts": {
            "confirmed": 0,
            "pending_camera": 1 if next_viewpoint_id is None else 0,
            "provisional": 2 if next_viewpoint_id is not None else 0,
            "rejected": 0,
        },
    }


def _completed() -> CompletedCoverageLeg:
    return CompletedCoverageLeg(Path("odom_execution_certificate.json"))


def _unexpected(name: str):
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError(f"unexpected call: {name}")

    return fail


def _admission(*, ready: bool = True):
    return SimpleNamespace(
        ready=ready,
        reasons=() if ready else ("planned_viewpoints_incomplete",),
        coverage_threshold_met=ready,
        visited_coverage_ratio=1.0 if ready else 0.5,
        visited_viewpoint_ids=("survey_vp_001",),
        planned_viewpoint_ids=("survey_vp_001",),
    )


def _pending_registry(count: int = 1):
    return SimpleNamespace(
        candidates=tuple(
            SimpleNamespace(
                candidate_uid=f"candidate_{index}",
                status=STATUS_PENDING_CAMERA,
            )
            for index in range(count)
        )
    )


class AutonomousCoverageMissionTest(unittest.TestCase):
    def test_completed_leg_requires_odom_execution_certificate(self):
        with self.assertRaises(ValueError):
            CompletedCoverageLeg(None)

    def test_one_completed_leg_captures_and_fuses_exactly_once_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), leg_limit=1)
            events: list[str] = []
            requests: dict[str, object] = {}

            def execute(request):
                events.append("execute")
                requests["leg"] = request
                return _completed()

            def capture(viewpoint_id, certificate_path):
                events.append("capture")
                requests["observation"] = (viewpoint_id, certificate_path)
                return Path("observer_summary.json")

            def fuse(viewpoint_id, observer_summary_path):
                events.append("fuse")
                requests["fusion"] = (viewpoint_id, observer_summary_path)
                return _summary("survey_vp_002", visited=1)

            def publish(request):
                events.append("checkpoint")
                requests["checkpoint"] = request
                return PublishedCoverageCheckpoint(
                    Path("checkpoint_001.json"), HASH_C
                )

            read_count = 0

            def read_summary(path):
                nonlocal read_count
                self.assertEqual(path, config.survey_root / "survey_summary.json")
                read_count += 1
                return _summary("survey_vp_001", visited=0)

            outcome = execute_coverage_mission(
                config,
                CoverageMissionEffects(
                    execute_completed_leg=execute,
                    capture_lidar_epoch=capture,
                    fuse_coverage_stop=fuse,
                    build_snapshot=_unexpected("build_snapshot"),
                    read_summary=read_summary,
                    publish_checkpoint=publish,
                    load_progress=_unexpected("load_progress"),
                    load_registry=_unexpected("load_registry"),
                ),
            )

            self.assertIsInstance(outcome, CoverageCheckpointComplete)
            self.assertEqual(events, ["execute", "capture", "fuse", "checkpoint"])
            self.assertEqual(read_count, 1)
            self.assertEqual(requests["leg"].target_viewpoint_id, "survey_vp_001")
            self.assertEqual(
                requests["observation"][1],
                Path("odom_execution_certificate.json"),
            )
            self.assertEqual(
                requests["fusion"][1],
                Path("observer_summary.json"),
            )

    def test_checkpoint_outcome_is_non_authorizing_and_operator_facing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), leg_limit=1)
            effects = CoverageMissionEffects(
                execute_completed_leg=lambda request: _completed(),
                capture_lidar_epoch=lambda viewpoint_id, certificate: Path(
                    "observer.json"
                ),
                fuse_coverage_stop=lambda viewpoint_id, observer: _summary(
                    "survey_vp_002", visited=1
                ),
                build_snapshot=_unexpected("build_snapshot"),
                read_summary=lambda path: _summary("survey_vp_001", visited=0),
                publish_checkpoint=lambda request: PublishedCoverageCheckpoint(
                    Path("manifest.json"), HASH_C
                ),
                load_progress=_unexpected("load_progress"),
                load_registry=_unexpected("load_registry"),
            )

            outcome = execute_coverage_mission(config, effects)
            summary = outcome.to_mission_summary()

            self.assertFalse(outcome.motion_authorized)
            self.assertTrue(summary["motion_published"])
            self.assertTrue(summary["prior_leg_motion_published"])
            self.assertFalse(summary["checkpoint_motion_authorized"])
            self.assertFalse(summary["coverage_complete"])
            self.assertAlmostEqual(summary["visited_coverage_ratio"], 1 / 3)
            self.assertEqual(summary["candidate_counts"]["provisional"], 2)
            self.assertEqual(summary["next_required_action"], RESUME_FROM_CHECKPOINT)
            with self.assertRaises(FrozenInstanceError):
                outcome.completed_coverage_legs = 99

    def test_final_leg_runs_admission_and_snapshot_even_at_leg_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), leg_limit=1)
            events: list[str] = []
            registry = _pending_registry()
            snapshot = SimpleNamespace(candidates=registry.candidates)

            def event(name, value):
                def callback(*args):
                    del args
                    events.append(name)
                    return value

                return callback

            outcome = execute_coverage_mission(
                config,
                CoverageMissionEffects(
                    execute_completed_leg=event("execute", _completed()),
                    capture_lidar_epoch=event("capture", Path("observer.json")),
                    fuse_coverage_stop=event("fuse", _summary(None, visited=1, total=1)),
                    read_summary=lambda path: _summary(
                        "survey_vp_001", visited=0, total=1
                    ),
                    publish_checkpoint=_unexpected("publish_checkpoint"),
                    load_progress=event("load_progress", object()),
                    load_registry=event("load_registry", registry),
                    evaluate_admission=event("admit", _admission()),
                    write_admission=event("write_admission", HASH_A),
                    build_snapshot=event("build_snapshot", snapshot),
                    write_snapshot=event("write_snapshot", HASH_B),
                    snapshot_sha256=event("snapshot_sha256", HASH_B),
                ),
            )

            self.assertIsInstance(outcome, CoverageComplete)
            self.assertEqual(
                events,
                [
                    "execute",
                    "capture",
                    "fuse",
                    "load_progress",
                    "load_registry",
                    "admit",
                    "write_admission",
                    "build_snapshot",
                    "write_snapshot",
                    "snapshot_sha256",
                ],
            )
            self.assertEqual(outcome.stand_count, 1)
            self.assertTrue(outcome.coverage_status.coverage_complete)
            self.assertEqual(
                outcome.coverage_status.next_required_action,
                CANDIDATE_SNAPSHOT_READY,
            )

    def test_observation_or_fusion_failure_never_publishes_checkpoint(self):
        for failure_stage in ("capture", "fuse"):
            with self.subTest(failure_stage=failure_stage):
                with tempfile.TemporaryDirectory() as tmp:
                    config = _config(Path(tmp), leg_limit=1)
                    events: list[str] = []

                    def execute(request):
                        del request
                        events.append("execute")
                        return _completed()

                    def capture(viewpoint_id, certificate):
                        del viewpoint_id, certificate
                        events.append("capture")
                        if failure_stage == "capture":
                            raise RuntimeError("capture failed")
                        return Path("observer.json")

                    def fuse(viewpoint_id, observer):
                        del viewpoint_id, observer
                        events.append("fuse")
                        raise RuntimeError("fusion failed")

                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        execute_coverage_mission(
                            config,
                            CoverageMissionEffects(
                                execute_completed_leg=execute,
                                capture_lidar_epoch=capture,
                                fuse_coverage_stop=fuse,
                                build_snapshot=_unexpected("build_snapshot"),
                                read_summary=lambda path: _summary(
                                    "survey_vp_001", visited=0
                                ),
                                publish_checkpoint=_unexpected(
                                    "publish_checkpoint"
                                ),
                                load_progress=_unexpected("load_progress"),
                                load_registry=_unexpected("load_registry"),
                            ),
                        )
                    expected = ["execute", "capture"]
                    if failure_stage == "fuse":
                        expected.append("fuse")
                    self.assertEqual(events, expected)

    def test_admission_and_expected_count_both_precede_snapshot(self):
        cases = (
            (False, 1, "coverage candidate admission rejected"),
            (True, 0, "did not resolve the expected stand count"),
        )
        for ready, count, error_code in cases:
            with self.subTest(ready=ready, count=count):
                with tempfile.TemporaryDirectory() as tmp:
                    config = _config(
                        Path(tmp), leg_limit=1, expected_stand_count=1
                    )
                    registry = _pending_registry(count)
                    with self.assertRaisesRegex(RuntimeError, error_code):
                        execute_coverage_mission(
                            config,
                            CoverageMissionEffects(
                                execute_completed_leg=lambda request: _completed(),
                                capture_lidar_epoch=lambda viewpoint_id, certificate: Path(
                                    "observer.json"
                                ),
                                fuse_coverage_stop=lambda viewpoint_id, observer: _summary(
                                    None, visited=1, total=1
                                ),
                                build_snapshot=_unexpected("build_snapshot"),
                                read_summary=lambda path: _summary(
                                    "survey_vp_001", visited=0, total=1
                                ),
                                publish_checkpoint=_unexpected(
                                    "publish_checkpoint"
                                ),
                                load_progress=lambda path, plan: object(),
                                load_registry=lambda path, plan: registry,
                                evaluate_admission=lambda plan, progress, registry: _admission(
                                    ready=ready
                                ),
                                write_admission=lambda path, decision: HASH_A,
                                write_snapshot=_unexpected("write_snapshot"),
                            ),
                        )

    def test_resume_limit_counts_this_run_while_cursor_reports_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent_manifest.json"
            config = _config(
                root,
                leg_limit=1,
                initial_leg_index=4,
                parent_checkpoint=parent,
            )
            checkpoints = []
            next_by_target = {
                "survey_vp_001": "survey_vp_002",
                "survey_vp_002": "survey_vp_003",
            }

            def fuse(viewpoint_id, observer):
                del observer
                target_index = int(viewpoint_id.rsplit("_", 1)[1])
                return _summary(
                    next_by_target[viewpoint_id],
                    visited=target_index,
                )

            def publish(request):
                checkpoints.append(request)
                return PublishedCoverageCheckpoint(
                    root / f"checkpoint_{len(checkpoints)}.json",
                    HASH_C,
                )

            outcome = execute_coverage_mission(
                config,
                CoverageMissionEffects(
                    execute_completed_leg=lambda request: _completed(),
                    capture_lidar_epoch=lambda viewpoint_id, certificate: Path(
                        f"{viewpoint_id}_observer.json"
                    ),
                    fuse_coverage_stop=fuse,
                    build_snapshot=_unexpected("build_snapshot"),
                    read_summary=lambda path: _summary(
                        "survey_vp_001", visited=0
                    ),
                    publish_checkpoint=publish,
                    load_progress=_unexpected("load_progress"),
                    load_registry=_unexpected("load_registry"),
                ),
            )

            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(checkpoints[0].completed_coverage_legs, 5)
            self.assertEqual(checkpoints[0].parent_checkpoint_path, parent)
            self.assertIs(
                checkpoints[0].identity,
                config.checkpoint_identity,
            )
            self.assertEqual(outcome.parent_checkpoint_path, parent)
            self.assertEqual(outcome.legs_completed_this_run, 1)

    def test_multi_leg_checkpoint_chain_uses_prior_published_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent_manifest.json"
            config = _config(root, leg_limit=2, parent_checkpoint=parent)
            checkpoints = []

            def fuse(viewpoint_id, observer):
                del observer
                if viewpoint_id == "survey_vp_001":
                    return _summary("survey_vp_002", visited=1)
                return _summary("survey_vp_003", visited=2)

            def publish(request):
                checkpoints.append(request)
                return PublishedCoverageCheckpoint(
                    root / f"checkpoint_{len(checkpoints)}.json",
                    HASH_C,
                )

            outcome = execute_coverage_mission(
                config,
                CoverageMissionEffects(
                    execute_completed_leg=lambda request: _completed(),
                    capture_lidar_epoch=lambda viewpoint_id, certificate: Path(
                        "observer.json"
                    ),
                    fuse_coverage_stop=fuse,
                    build_snapshot=_unexpected("build_snapshot"),
                    read_summary=lambda path: _summary(
                        "survey_vp_001", visited=0
                    ),
                    publish_checkpoint=publish,
                    load_progress=_unexpected("load_progress"),
                    load_registry=_unexpected("load_registry"),
                ),
            )

            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(checkpoints[0].parent_checkpoint_path, parent)
            self.assertEqual(
                checkpoints[1].parent_checkpoint_path,
                root / "checkpoint_1.json",
            )
            self.assertIs(
                checkpoints[0].identity,
                checkpoints[1].identity,
            )
            self.assertEqual(
                outcome.checkpoint_manifest,
                root / "checkpoint_2.json",
            )

    def test_module_has_no_parent_runner_ros_subprocess_or_prompt_import(self):
        source = inspect.getsource(mission)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

        self.assertFalse(
            any(
                name.endswith("run_autonomous_stand_exploration")
                for name in imported
            )
        )
        self.assertFalse(any(name in {"rclpy", "subprocess"} for name in imported))
        self.assertNotIn("input(", source)


if __name__ == "__main__":
    unittest.main()
