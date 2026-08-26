import unittest

from scripts.aufgabe04.real_robot.mission.modes import (
    AutonomousAuthorizationScope,
    AutonomousRunMode,
    resolve_autonomous_run_mode,
    validate_autonomous_viewpoint_scope,
    validate_session_id_mode_label,
)


class AutonomousRunModeTests(unittest.TestCase):
    def test_default_is_safe_first_leg_dry_run(self):
        resolved = resolve_autonomous_run_mode()

        self.assertEqual(resolved.mode, AutonomousRunMode.DRY_FIRST_LEG)
        self.assertFalse(resolved.execute)
        self.assertEqual(resolved.coverage_leg_limit, 0)
        self.assertFalse(resolved.stop_after_coverage)
        self.assertEqual(
            resolved.authorization_scope, AutonomousAuthorizationScope.NONE
        )
        self.assertIn("no physical motion", resolved.authorization_scope_text)

    def test_legacy_execution_settings_resolve_deterministically(self):
        cases = (
            (
                {"execute": True, "coverage_leg_limit": 2},
                AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT,
                AutonomousAuthorizationScope.BOUNDED_COVERAGE,
            ),
            (
                {"execute": True, "stop_after_coverage": True},
                AutonomousRunMode.EXECUTE_COVERAGE_ONLY,
                AutonomousAuthorizationScope.COVERAGE_ONLY,
            ),
            (
                {"execute": True},
                AutonomousRunMode.EXECUTE_FULL,
                AutonomousAuthorizationScope.FULL_MISSION,
            ),
        )
        for kwargs, expected_mode, expected_scope in cases:
            with self.subTest(kwargs=kwargs):
                resolved = resolve_autonomous_run_mode(**kwargs)
                self.assertEqual(resolved.mode, expected_mode)
                self.assertEqual(resolved.authorization_scope, expected_scope)
                self.assertTrue(resolved.execute)

    def test_explicit_execution_modes_replace_execute_flag(self):
        full = resolve_autonomous_run_mode(run_mode="execute-full")
        coverage = resolve_autonomous_run_mode(
            run_mode=AutonomousRunMode.EXECUTE_COVERAGE_ONLY
        )

        self.assertTrue(full.execute)
        self.assertEqual(full.mode, AutonomousRunMode.EXECUTE_FULL)
        self.assertTrue(coverage.execute)
        self.assertTrue(coverage.stop_after_coverage)

    def test_exact_two_camera_has_fixed_scope_and_camera_authority(self):
        for coverage_leg_limit in (0, 2):
            with self.subTest(coverage_leg_limit=coverage_leg_limit):
                resolved = resolve_autonomous_run_mode(
                    run_mode="execute-exact-two-camera",
                    coverage_leg_limit=coverage_leg_limit,
                )

                self.assertTrue(resolved.execute)
                self.assertEqual(
                    resolved.mode,
                    AutonomousRunMode.EXECUTE_EXACT_TWO_CAMERA,
                )
                self.assertEqual(resolved.coverage_leg_limit, 2)
                self.assertFalse(resolved.stop_after_coverage)
                self.assertTrue(resolved.camera_phase_enabled)
                self.assertEqual(
                    resolved.authorization_scope,
                    AutonomousAuthorizationScope.EXACT_TWO_CAMERA,
                )
                self.assertIn(
                    "camera-approach phase",
                    resolved.authorization_scope_text,
                )

        self.assertTrue(
            resolve_autonomous_run_mode(
                run_mode="execute-full"
            ).camera_phase_enabled
        )
        for mode in (
            "dry-first-leg",
            "execute-coverage-checkpoint",
            "execute-coverage-only",
            "resume-next-coverage-leg",
        ):
            kwargs = (
                {"coverage_leg_limit": 1}
                if mode == "execute-coverage-checkpoint"
                else {}
            )
            with self.subTest(mode=mode):
                self.assertFalse(
                    resolve_autonomous_run_mode(
                        run_mode=mode,
                        **kwargs,
                    ).camera_phase_enabled
                )

    def test_exact_two_camera_rejects_scope_contradictions(self):
        for kwargs, message in (
            (
                {
                    "run_mode": "execute-exact-two-camera",
                    "coverage_leg_limit": 1,
                },
                "fixed two-leg",
            ),
            (
                {
                    "run_mode": "execute-exact-two-camera",
                    "coverage_leg_limit": 3,
                },
                "fixed two-leg",
            ),
            (
                {
                    "run_mode": "execute-exact-two-camera",
                    "stop_after_coverage": True,
                },
                "requires the camera phase",
            ),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                resolve_autonomous_run_mode(**kwargs)

    def test_resume_mode_has_fixed_one_leg_non_authorizing_intent(self):
        resolved = resolve_autonomous_run_mode(
            run_mode="resume-next-coverage-leg"
        )

        self.assertTrue(resolved.execute)
        self.assertEqual(resolved.coverage_leg_limit, 1)
        self.assertFalse(resolved.stop_after_coverage)
        self.assertIn("exactly one next", resolved.authorization_scope_text)
        with self.assertRaisesRegex(ValueError, "owns its one-leg"):
            resolve_autonomous_run_mode(
                run_mode="resume-next-coverage-leg",
                coverage_leg_limit=2,
            )

    def test_explicit_checkpoint_requires_and_preserves_positive_limit(self):
        resolved = resolve_autonomous_run_mode(
            run_mode="execute-coverage-checkpoint",
            coverage_leg_limit=3,
        )

        self.assertEqual(
            resolved.mode, AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT
        )
        self.assertEqual(resolved.coverage_leg_limit, 3)
        self.assertEqual(
            resolved.authorization_scope_text,
            "at most 3 center-corridor coverage leg(s)",
        )

    def test_redundant_compatible_legacy_flags_are_accepted(self):
        full = resolve_autonomous_run_mode(
            run_mode="execute-full", execute=True
        )
        coverage = resolve_autonomous_run_mode(
            run_mode="execute-coverage-only",
            execute=True,
            stop_after_coverage=True,
        )
        checkpoint = resolve_autonomous_run_mode(
            run_mode="execute-coverage-checkpoint",
            execute=True,
            coverage_leg_limit=1,
        )

        self.assertEqual(full.mode, AutonomousRunMode.EXECUTE_FULL)
        self.assertEqual(
            coverage.mode, AutonomousRunMode.EXECUTE_COVERAGE_ONLY
        )
        self.assertEqual(
            checkpoint.mode, AutonomousRunMode.EXECUTE_COVERAGE_CHECKPOINT
        )

    def test_legacy_checkpoint_selectors_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError, "different execution checkpoints"):
            resolve_autonomous_run_mode(
                execute=True,
                coverage_leg_limit=1,
                stop_after_coverage=True,
            )

    def test_dry_legacy_mode_rejects_execution_only_options(self):
        with self.assertRaisesRegex(ValueError, "requires an execution mode"):
            resolve_autonomous_run_mode(coverage_leg_limit=1)
        with self.assertRaisesRegex(ValueError, "requires an execution mode"):
            resolve_autonomous_run_mode(stop_after_coverage=True)

    def test_explicit_modes_reject_contradictory_legacy_options(self):
        cases = (
            (
                {"run_mode": "dry-first-leg", "execute": True},
                "contradicts legacy physical-execution",
            ),
            (
                {
                    "run_mode": "execute-coverage-checkpoint",
                    "coverage_leg_limit": 1,
                    "stop_after_coverage": True,
                },
                "contradicts stop_after_coverage",
            ),
            (
                {
                    "run_mode": "execute-coverage-only",
                    "coverage_leg_limit": 1,
                },
                "contradicts a positive coverage_leg_limit",
            ),
            (
                {
                    "run_mode": "execute-full",
                    "stop_after_coverage": True,
                },
                "contradicts stop_after_coverage",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_autonomous_run_mode(**kwargs)

    def test_checkpoint_mode_rejects_zero_limit(self):
        with self.assertRaisesRegex(ValueError, "requires a positive"):
            resolve_autonomous_run_mode(
                run_mode="execute-coverage-checkpoint"
            )

    def test_invalid_modes_and_legacy_types_fail_closed(self):
        invalid_calls = (
            ({"run_mode": "fast-and-loose"}, "unknown autonomous run mode"),
            ({"run_mode": 1}, "run_mode must be"),
            ({"execute": 1}, "execute must be a boolean"),
            (
                {"stop_after_coverage": 1},
                "stop_after_coverage must be a boolean",
            ),
            (
                {"coverage_leg_limit": True},
                "coverage_leg_limit must be an integer",
            ),
            (
                {"coverage_leg_limit": 1.5},
                "coverage_leg_limit must be an integer",
            ),
            (
                {"coverage_leg_limit": -1},
                "coverage_leg_limit must be non-negative",
            ),
        )
        for kwargs, message in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_autonomous_run_mode(**kwargs)

    def test_session_label_cannot_contradict_resolved_mode(self):
        execute = resolve_autonomous_run_mode(run_mode="execute-full")
        dry = resolve_autonomous_run_mode()

        validate_session_id_mode_label("parkour_execute_001", execute)
        validate_session_id_mode_label("parkour_dry_001", dry)
        with self.assertRaisesRegex(ValueError, "must not be labelled as dry"):
            validate_session_id_mode_label("parkour_dry_001", execute)
        with self.assertRaisesRegex(ValueError, "must not be labelled as execute"):
            validate_session_id_mode_label("parkour_execute_001", dry)

    def test_session_id_rejects_path_traversal_and_unsafe_characters(self):
        dry = resolve_autonomous_run_mode()

        for session_id in (
            "../outside",
            "/absolute",
            "nested/session",
            "white space",
            "",
            "x" * 129,
        ):
            with self.subTest(session_id=session_id), self.assertRaisesRegex(
                ValueError,
                "safe 1-128 character identifier",
            ):
                validate_session_id_mode_label(session_id, dry)

    def test_exact_two_is_rejected_for_complete_coverage_modes(self):
        for mode in ("execute-coverage-only", "execute-full"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError,
                "diagnostic-only",
            ):
                validate_autonomous_viewpoint_scope(
                    resolve_autonomous_run_mode(run_mode=mode),
                    exact_inspection_point_count=2,
                )

    def test_exact_two_camera_requires_exactly_two_inspection_points(self):
        resolved = resolve_autonomous_run_mode(
            run_mode="execute-exact-two-camera"
        )

        validate_autonomous_viewpoint_scope(
            resolved,
            exact_inspection_point_count=2,
        )
        for count in (None, True, 1, 2.0, 3):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError,
                "requires --exact-inspection-point-count 2",
            ):
                validate_autonomous_viewpoint_scope(
                    resolved,
                    exact_inspection_point_count=count,
                )

    def test_exact_two_remains_available_to_bounded_diagnostic_modes(self):
        modes = (
            resolve_autonomous_run_mode(run_mode="dry-first-leg"),
            resolve_autonomous_run_mode(
                run_mode="execute-coverage-checkpoint",
                coverage_leg_limit=1,
            ),
        )
        for resolved in modes:
            with self.subTest(mode=resolved.mode.value):
                validate_autonomous_viewpoint_scope(
                    resolved,
                    exact_inspection_point_count=2,
                )

        validate_autonomous_viewpoint_scope(
            resolve_autonomous_run_mode(run_mode="execute-full"),
            exact_inspection_point_count=None,
        )


if __name__ == "__main__":
    unittest.main()
