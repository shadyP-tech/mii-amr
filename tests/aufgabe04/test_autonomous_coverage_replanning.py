import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.aufgabe04.navigation.models import Pose2D
from scripts.aufgabe04.real_robot.autonomous_child_runner import MotionLegOutcome
from scripts.aufgabe04.real_robot import autonomous_coverage_replanning as replanning
from scripts.aufgabe04.real_robot import autonomous_coverage_execution as execution
from scripts.aufgabe04.real_robot.autonomous_coverage_execution import (
    CoverageLegEffects,
)


def _outcome(
    root: Path,
    *,
    run_id: str = "coverage_000",
    status: str = "stopped",
    stop_reason: str = "",
    stop_details: dict[str, object] | None = None,
    motion_published: bool = False,
    semantic_log_start_offset: int = 0,
) -> MotionLegOutcome:
    semantic_log = root / "run.jsonl"
    if not semantic_log.exists():
        semantic_log.write_text("{}\n")
    return MotionLegOutcome(
        run_id=run_id,
        status=status,
        stop_reason=stop_reason,
        stop_details={} if stop_details is None else stop_details,
        motion_published=motion_published,
        returncode=1,
        semantic_log_path=semantic_log,
        semantic_log_start_offset=semantic_log_start_offset,
    )


def _startup_outcome(root: Path, *, route_pose: object) -> MotionLegOutcome:
    return _outcome(
        root,
        stop_reason="pose outside certified startup segment",
        stop_details={
            "source": "execution_route_certificate",
            "phase": "before_motion_confirmation",
            "route_pose": route_pose,
        },
    )


def _runtime_localization_details() -> dict[str, object]:
    return {
        "fault_code": "localization_reseal_required",
        "source": "global_consistency_monitor",
        "execution_pose_owner": "odom",
        "global_consistency_monitor": "amcl",
        "monitor_action": "FORCE_ZERO_RESEAL",
        "fail_closed": True,
        "continuity": {
            "accepted": False,
            "requires_zero_cycle": True,
            "requires_reseal": True,
            "decision": "force_zero_reseal",
            "reason": "map_from_odom_yaw_drift",
            "fail_closed": True,
        },
    }


class AutonomousCoverageReplanningTest(unittest.TestCase):
    def test_public_api_contains_no_private_helpers(self):
        expected = {
            "adopted_blockage_replans_for_run",
            "advance_transient_overlay_resume_state",
            "coverage_reseal_suffix",
            "is_resealable_startup_mismatch",
            "is_runtime_localization_reseal_required",
            "load_coverage_plan",
            "replan_runtime_localization_source",
            "replan_source_preserving_transient_overlay",
            "replan_startup_source",
            "startup_reseal_pose",
        }
        self.assertEqual(set(replanning.__all__), expected)
        self.assertTrue(all(not name.startswith("_") for name in replanning.__all__))
        self.assertTrue(hasattr(replanning, "_replan_coverage_source_from_pose"))
        self.assertNotIn("_replan_coverage_source_from_pose", replanning.__all__)

    def test_executor_exports_only_its_public_orchestration_contract(self):
        self.assertEqual(
            execution.__all__,
            [
                "CoverageLegConfig",
                "CoverageLegEffects",
                "MissionLegPermitContext",
                "RuntimeLocalizationPermitContext",
                "execute_coverage_leg_with_replans",
            ],
        )
        moved_private_names = (
            "_adopted_blockage_replans_for_run",
            "_advance_transient_overlay_resume_state",
            "_coverage_reseal_suffix",
            "_execute_coverage_leg_with_replans",
            "_is_resealable_startup_mismatch",
            "_is_runtime_localization_reseal_required",
            "_replan_coverage_source_from_pose",
            "_replan_runtime_localization_source",
            "_replan_source_preserving_transient_overlay",
            "_replan_startup_source",
            "_startup_reseal_pose",
        )
        for name in moved_private_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(execution, name))

    def test_module_has_no_parent_ros_process_or_prompt_import(self):
        source = Path(replanning.__file__).read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            any("run_autonomous_stand_exploration" in name for name in imported)
        )
        self.assertNotIn("subprocess", imported)
        self.assertNotIn("rclpy", imported)
        self.assertNotIn("rospy", imported)
        loaded_names = {
            name.id for name in ast.walk(tree) if isinstance(name, ast.Name)
        }
        self.assertNotIn("input", loaded_names)

    def test_reseal_suffix_keeps_attempt_identities_distinct(self):
        self.assertEqual(
            replanning.coverage_reseal_suffix(
                startup_reseal_index=0,
                runtime_localization_reseal_index=0,
            ),
            "",
        )
        self.assertEqual(
            replanning.coverage_reseal_suffix(
                startup_reseal_index=2,
                runtime_localization_reseal_index=3,
            ),
            "_startup_reseal_002_runtime_localization_reseal_003",
        )
        with self.assertRaises(ValueError):
            replanning.coverage_reseal_suffix(
                startup_reseal_index=-1,
                runtime_localization_reseal_index=0,
            )

    def test_startup_classifier_and_pose_reject_motion_or_nonfinite_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome = _startup_outcome(
                root,
                route_pose={"x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.5},
            )
            self.assertTrue(replanning.is_resealable_startup_mismatch(outcome))
            self.assertEqual(
                replanning.startup_reseal_pose(outcome),
                Pose2D(1.0, 2.0, 0.5),
            )
            published = MotionLegOutcome(
                **{**outcome.__dict__, "motion_published": True}
            )
            self.assertFalse(
                replanning.is_resealable_startup_mismatch(published)
            )
            nonfinite = _startup_outcome(
                root,
                route_pose={"x_m": float("nan"), "y_m": 2.0, "yaw_rad": 0.5},
            )
            with self.assertRaisesRegex(ValueError, "must be finite"):
                replanning.startup_reseal_pose(nonfinite)

    def test_runtime_classifier_requires_complete_post_motion_stop_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eligible = _outcome(
                root,
                stop_details=_runtime_localization_details(),
                motion_published=True,
            )
            self.assertTrue(
                replanning.is_runtime_localization_reseal_required(eligible)
            )
            no_motion = MotionLegOutcome(
                **{**eligible.__dict__, "motion_published": False}
            )
            self.assertFalse(
                replanning.is_runtime_localization_reseal_required(no_motion)
            )

    def test_adopted_events_are_scoped_to_post_admission_log_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.jsonl"
            prior = json.dumps(
                {
                    "event": "transient_navigation_blockage_replanned",
                    "run_id": "wrong_prior_run",
                },
                separators=(",", ":"),
            ) + "\n"
            valid = {
                "event": "transient_navigation_blockage_replanned",
                "run_id": "coverage_000",
                "post_plan_runtime_revalidated": True,
                "semantic_survey_evidence": False,
            }
            log.write_text(
                prior
                + json.dumps(valid, separators=(",", ":"))
                + "\n"
                + json.dumps({"event": "motion_started"})
                + "\n"
            )

            adopted = replanning.adopted_blockage_replans_for_run(
                log,
                run_id="coverage_000",
                start_offset=len(prior.encode("utf-8")),
            )

            self.assertEqual(adopted, [valid])

    def test_adopted_event_validation_fails_closed(self):
        invalid_payloads = (
            {
                "run_id": "other_run",
                "post_plan_runtime_revalidated": True,
                "semantic_survey_evidence": False,
            },
            {
                "run_id": "coverage_000",
                "post_plan_runtime_revalidated": False,
                "semantic_survey_evidence": False,
            },
            {
                "run_id": "coverage_000",
                "post_plan_runtime_revalidated": True,
                "semantic_survey_evidence": True,
            },
        )
        for fields in invalid_payloads:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "run.jsonl"
                log.write_text(
                    json.dumps(
                        {
                            "event": "transient_navigation_blockage_replanned",
                            **fields,
                        }
                    )
                    + "\n"
                )
                with self.assertRaises(RuntimeError):
                    replanning.adopted_blockage_replans_for_run(
                        log,
                        run_id="coverage_000",
                    )

    def test_uncertainty_branch_refuses_unadmitted_overlay_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome = _outcome(root)
            event = {
                "event": "transient_navigation_blockage_replanned",
                "run_id": outcome.run_id,
                "post_plan_runtime_revalidated": True,
                "semantic_survey_evidence": False,
            }
            with (
                patch.object(
                    replanning,
                    "adopted_blockage_replans_for_run",
                    return_value=[event],
                ),
                patch.object(replanning, "load_coverage_survey_plan"),
                patch.object(
                    replanning,
                    "update_transient_overlay_resume_state_from_events",
                ) as update,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "lacks accepted uncertainty evidence",
                ):
                    replanning.advance_transient_overlay_resume_state(
                        outcome=outcome,
                        previous_state=None,
                        plan_path=root / "plan.json",
                        leg_index=0,
                        target_viewpoint_id="survey_vp_001",
                        max_replans=2,
                        require_uncertainty_admission=True,
                        artifact_root=root,
                        survey_root=root / "survey",
                    )
                update.assert_not_called()

    def test_public_replan_wrappers_bind_recovery_kind_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcome = _outcome(root)
            common = Mock(return_value={"route_csv": "route.csv"})
            base = {
                "map_yaml": root / "map.yaml",
                "semantic_map_id": "arena",
                "survey_root": root / "survey",
                "plan_path": root / "plan.json",
                "expected_target_viewpoint_id": "survey_vp_001",
                "current_pose": Pose2D(0.0, 0.0, 0.0),
                "rejected_outcome": outcome,
                "reseal_index": 1,
                "output_dir": root / "replan",
            }
            with patch.object(
                replanning,
                "_replan_coverage_source_from_pose",
                common,
            ):
                replanning.replan_startup_source(**base)
                replanning.replan_runtime_localization_source(**base)
            self.assertEqual(
                common.call_args_list[0].kwargs["reseal_kind"],
                "startup",
            )
            self.assertEqual(
                common.call_args_list[0].kwargs["status"],
                "startup_route_replanned",
            )
            self.assertEqual(
                common.call_args_list[1].kwargs["reseal_kind"],
                "runtime_localization",
            )
            self.assertEqual(
                common.call_args_list[1].kwargs["status"],
                "runtime_localization_route_replanned",
            )

    def test_coverage_effect_defaults_resolve_public_replanner_late(self):
        replacement = Mock(return_value={"route_csv": "replacement.csv"})
        effects = CoverageLegEffects(
            run_motion_leg=Mock(),
            admit_preplanning_localization=Mock(),
        )

        with patch.object(replanning, "replan_startup_source", replacement):
            result = effects.replan_startup_source(marker="late-bound")

        self.assertEqual(result, {"route_csv": "replacement.csv"})
        replacement.assert_called_once_with(marker="late-bound")


if __name__ == "__main__":
    unittest.main()
