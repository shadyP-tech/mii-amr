import unittest

from scripts.aufgabe04.real_robot.readiness.initialpose_prompt import (
    InitialPosePromptConfig,
    prepare_preplanning_initialpose,
    prompt_for_initialpose_attempt,
)


class InitialPosePromptTest(unittest.TestCase):
    def test_prompt_mentions_preplanning_no_motion_and_expected_click_window(self):
        lines: list[str] = []
        prompts: list[str] = []
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.5,
            maximum_retry_count=0,
        )

        prompt_for_initialpose_attempt(
            config=config,
            attempt_index=0,
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lines.append,
        )

        rendered = "\n".join(lines)
        self.assertIn("before preplanning localization", rendered)
        self.assertIn("Do not move the robot", rendered)
        self.assertIn("Do not send a Nav2 goal", rendered)
        self.assertIn("/amcl_pose", rendered)
        self.assertIn("2.5s", rendered)
        self.assertEqual(
            prompts,
            ["Press Enter, then click 2D Pose Estimate immediately: "],
        )

    def test_rejects_invalid_attempt_index(self):
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.0,
            maximum_retry_count=2,
        )

        with self.assertRaisesRegex(ValueError, "only before preplanning"):
            prompt_for_initialpose_attempt(
                config=config,
                attempt_index=1,
                input_fn=lambda _prompt: "",
                output_fn=lambda _line: None,
            )

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            InitialPosePromptConfig(
                amcl_topic="/amcl_pose",
                observation_window_sec=0.0,
                maximum_retry_count=1,
            )

    def test_preplanning_helper_prompts_once_when_enabled(self):
        lines: list[str] = []
        prompts: list[str] = []
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.0,
            maximum_retry_count=0,
        )

        prompted = prepare_preplanning_initialpose(
            enabled=True,
            config=config,
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lines.append,
        )

        self.assertTrue(prompted)
        self.assertIn("before preplanning localization", "\n".join(lines))
        self.assertEqual(
            prompts,
            ["Press Enter, then click 2D Pose Estimate immediately: "],
        )

    def test_preplanning_helper_skips_cleanly_when_disabled(self):
        lines: list[str] = []
        prompts: list[str] = []
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.0,
            maximum_retry_count=0,
        )

        prompted = prepare_preplanning_initialpose(
            enabled=False,
            config=config,
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lines.append,
        )

        self.assertFalse(prompted)
        self.assertEqual(lines, [])
        self.assertEqual(prompts, [])

    def test_preplanning_helper_disabled_skips_retry_config_validation(self):
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.0,
            maximum_retry_count=1,
        )

        prompted = prepare_preplanning_initialpose(
            enabled=False,
            config=config,
            input_fn=lambda _prompt: "",
            output_fn=lambda _line: None,
        )

        self.assertFalse(prompted)

    def test_preplanning_helper_rejects_retry_enabled_prompt_config(self):
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.0,
            maximum_retry_count=1,
        )

        with self.assertRaisesRegex(ValueError, "must disable prompt retries"):
            prepare_preplanning_initialpose(
                enabled=True,
                config=config,
                input_fn=lambda _prompt: "",
                output_fn=lambda _line: None,
            )


if __name__ == "__main__":
    unittest.main()
