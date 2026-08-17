import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "common" / "run_with_bundle.sh"


def write_executable(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text))
    path.chmod(0o755)


class RunWithBundleTest(unittest.TestCase):
    def make_fake_path(self, tmpdir: Path, *, ros2_exit: int = 0) -> str:
        fake_bin = tmpdir / "bin"
        fake_bin.mkdir(exist_ok=True)
        write_executable(
            fake_bin / "git",
            """\
            #!/usr/bin/env bash
            if [[ "$1" == "status" ]]; then
              echo " M fake_file"
            elif [[ "$1" == "rev-parse" && "$2" == "--abbrev-ref" ]]; then
              echo "main"
            elif [[ "$1" == "rev-parse" ]]; then
              echo "0123456789abcdef"
            else
              echo "git $*"
            fi
            """,
        )
        write_executable(
            fake_bin / "ros2",
            f"""\
            #!/usr/bin/env bash
            if [[ "${{ROS2_FAKE_DELAY_SEC:-}}" != "" && "$1 $2" == "topic echo" ]]; then
              echo "start $*" >>"${{ROS2_FAKE_EVENT_LOG}}"
              sleep "${{ROS2_FAKE_DELAY_SEC}}"
              echo "end $*" >>"${{ROS2_FAKE_EVENT_LOG}}"
            fi
            echo "ros2 $*"
            if [[ "${{ROS2_FAKE_LOG:-}}" != "" ]]; then
              echo "$*" >>"${{ROS2_FAKE_LOG}}"
            fi
            exit {ros2_exit}
            """,
        )
        return str(fake_bin) + os.pathsep + os.environ.get("PATH", "")

    def run_bundle(
        self,
        args,
        *,
        tmpdir: Path,
        ros2_exit: int = 0,
        ros2_delay_sec: float | None = None,
    ):
        env = os.environ.copy()
        env["PATH"] = self.make_fake_path(tmpdir, ros2_exit=ros2_exit)
        env["RUN_BUNDLE_ROOT"] = str(tmpdir / "bundles")
        env["ROS2_FAKE_LOG"] = str(tmpdir / "ros2_calls.txt")
        if ros2_delay_sec is not None:
            env["ROS2_FAKE_DELAY_SEC"] = str(ros2_delay_sec)
            env["ROS2_FAKE_EVENT_LOG"] = str(tmpdir / "ros2_events.txt")
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_creates_bundle_and_resolves_namespaced_topics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            result = self.run_bundle(
                [
                    "--namespace",
                    "robot1",
                    "--cmd-vel-topic",
                    "cmd_vel",
                    "--odom-topic",
                    "/odom",
                    "run_001",
                    "--",
                    sys.executable,
                    "-c",
                    "print('wrapped ok')",
                ],
                tmpdir=tmpdir,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = tmpdir / "bundles" / "run_001"
            self.assertTrue((bundle / "manifest.txt").exists())
            self.assertIn("wrapped ok", (bundle / "terminal_run.log").read_text())
            manifest = (bundle / "manifest.txt").read_text()
            self.assertIn("resolved_cmd_vel_topic=/robot1/cmd_vel", manifest)
            self.assertIn("resolved_odom_topic=/odom", manifest)
            self.assertIn("command_exit_code=0", manifest)
            for artifact_name in (
                "ros_topics.txt",
                "ros_nodes.txt",
                "ros_actions.txt",
                "cmd_vel_info.txt",
                "scan_once.txt",
                "odom_once.txt",
                "amcl_pose_once.txt",
                "navigate_to_pose_status_once.txt",
                "namespaced_navigate_to_pose_status_once.txt",
                "tf_frames.txt",
            ):
                self.assertTrue((bundle / artifact_name).exists(), artifact_name)
            ros2_calls = (tmpdir / "ros2_calls.txt").read_text()
            self.assertIn("topic info /robot1/cmd_vel --verbose", ros2_calls)
            self.assertIn("topic echo --once /robot1/scan", ros2_calls)

    def test_propagates_wrapped_command_exit_code_through_tee(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            result = self.run_bundle(
                ["run_fail", "--", sys.executable, "-c", "print('bad'); raise SystemExit(7)"],
                tmpdir=tmpdir,
            )

            self.assertEqual(result.returncode, 7)
            bundle = tmpdir / "bundles" / "run_fail"
            self.assertIn("bad", (bundle / "terminal_run.log").read_text())
            self.assertIn("command_exit_code=7", (bundle / "manifest.txt").read_text())
            self.assertTrue((bundle / "post_ros_topics.txt").exists())

    def test_exposes_exact_bundle_directory_to_wrapped_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            result = self.run_bundle(
                [
                    "run_trace",
                    "--",
                    sys.executable,
                    "-c",
                    (
                        "import os; from pathlib import Path; "
                        "root=Path(os.environ['MII_AMR_RUN_BUNDLE_DIR']); "
                        "(root/'controller_trace.jsonl').write_text('trace\\n'); "
                        "print(root)"
                    ),
                ],
                tmpdir=tmpdir,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = tmpdir / "bundles" / "run_trace"
            self.assertEqual(
                (bundle / "controller_trace.jsonl").read_text(),
                "trace\n",
            )
            self.assertIn(str(bundle), (bundle / "terminal_run.log").read_text())

    def test_rejects_unsafe_run_ids_and_missing_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad = self.run_bundle(["../bad", "--", "true"], tmpdir=tmpdir)
            missing = self.run_bundle(["run_001", "--"], tmpdir=tmpdir)

            self.assertEqual(bad.returncode, 2)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("COMMAND is required", missing.stderr)

    def test_best_effort_ros_failures_do_not_fail_successful_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            result = self.run_bundle(
                ["run_no_ros", "--", sys.executable, "-c", "print('still runs')"],
                tmpdir=tmpdir,
                ros2_exit=42,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = tmpdir / "bundles" / "run_no_ros"
            self.assertIn("still runs", (bundle / "terminal_run.log").read_text())
            self.assertIn("command failed with exit code 42", (bundle / "ros_topics.txt").read_text())

    def test_pre_run_topic_echo_captures_overlap_and_finish_before_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            started = time.monotonic()
            result = self.run_bundle(
                [
                    "--namespace",
                    "robot1",
                    "run_parallel",
                    "--",
                    sys.executable,
                    "-c",
                    (
                        "import os; from pathlib import Path; "
                        "events=Path(os.environ['ROS2_FAKE_EVENT_LOG']).read_text(); "
                        "assert events.count('end topic echo') == 5, events; "
                        "print('all pre-run captures complete')"
                    ),
                ],
                tmpdir=tmpdir,
                ros2_delay_sec=1.0,
            )
            elapsed = time.monotonic() - started

            self.assertEqual(result.returncode, 0, result.stderr)
            events = (tmpdir / "ros2_events.txt").read_text().splitlines()
            first_end = next(index for index, event in enumerate(events) if event.startswith("end "))
            self.assertGreaterEqual(
                sum(event.startswith("start ") for event in events[:first_end]),
                2,
                events,
            )
            # Five sequential one-second echoes would take at least five
            # seconds. The wider bound leaves ample scheduling headroom while
            # still catching a return to sequential collection.
            self.assertLess(elapsed, 3.5, (elapsed, events))
            terminal_log = tmpdir / "bundles" / "run_parallel" / "terminal_run.log"
            self.assertIn("all pre-run captures complete", terminal_log.read_text())


if __name__ == "__main__":
    unittest.main()
