"""The mission grants bounded geometry time and forwards explicit overrides."""

from unittest.mock import patch

import pytest

from scripts.aufgabe04.real_robot.autonomous_runner.cli import build_parser
from scripts.aufgabe04.real_robot.autonomous_runner import runtime
from tests.aufgabe04.test_autonomous_camera_capture import _args, _candidate, _write_measured_model


def parse(*extra):
    return build_parser().parse_args([
        "--robot-profile", "robot.json", "--camera-calibration", "camera.json",
        "--physical-site", "site.json", *extra])


def test_mission_grace_default_is_short_and_finite():
    assert parse().qr_pose_fallback_delay_sec == 1.5


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "10.1"])
def test_unbounded_or_negative_grace_is_rejected_before_launch(value):
    with pytest.raises(SystemExit):
        parse("--qr-pose-fallback-delay-sec", value)


@pytest.mark.parametrize("delay", [0., .75, 10.])
def test_explicit_grace_reaches_observer_without_changing_camera_deadline(tmp_path, delay):
    args = _args(_write_measured_model(tmp_path))
    args.qr_pose_fallback_delay_sec = parse("--qr-pose-fallback-delay-sec", str(delay)).qr_pose_fallback_delay_sec
    with patch.object(runtime.subprocess, "Popen") as popen, patch.object(
            runtime, "monitor_passive_observer_process", side_effect=RuntimeError("stop after launch")) as monitor:
        with pytest.raises(RuntimeError, match="stop after launch"):
            runtime._capture_camera_recommendation(profile=object(), args=args,
                candidate=_candidate(), output_dir=tmp_path / "attempt")
    command = popen.call_args.args[0]
    assert command[command.index("--qr-pose-fallback-delay-sec") + 1] == str(delay)
    assert monitor.call_args.kwargs["timeout_sec"] == 90.
