import unittest

from scripts.aufgabe04.real_robot.readiness.initialpose_prompt import (
    InitialPosePromptConfig,
    prompt_for_initialpose_attempt,
)


class InitialPosePromptTest(unittest.TestCase):
    def test_prompt_mentions_no_motion_and_expected_click_window(self):
        lines: list[str] = []
        prompts: list[str] = []
        config = InitialPosePromptConfig(
            amcl_topic="/amcl_pose",
            observation_window_sec=2.5,
            maximum_retry_count=2,
        )

        prompt_for_initialpose_attempt(
            config=config,
            attempt_index=1,
            input_fn=lambda prompt: prompts.append(prompt) or "",
            output_fn=lines.append,
        )

        rendered = "\n".join(lines)
        self.assertIn("readiness retry", rendered)
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
            maximum_retry_count=1,
        )

        with self.assertRaises(ValueError):
            prompt_for_initialpose_attempt(
                config=config,
                attempt_index=2,
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


if __name__ == "__main__":
    unittest.main()
