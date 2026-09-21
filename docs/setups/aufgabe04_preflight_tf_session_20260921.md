# Persistent preflight TF and bounded snapshot admission

The autonomous mission now owns one observation-only preflight service across
its planning, dry-run, execute, and recovery subprocesses. This keeps the TF
listener and buffer warm between independently certified legs. Each request
still constructs and destroys a fresh `RosPreflightNode`, including its direct
`/tf` capture, AMCL samples, sensor receipts, and nomotion-update client.

## Modules

- `navigation/localization/preflight_session.py`: private Unix socket, bounded
  JSON messages, client result validation, service readiness handshake, and
  mission-scoped process cleanup. Child processes inherit the socket through
  `AUFGABE04_PREFLIGHT_SESSION_SOCKET`. Requests are serialized. A configured
  service failure rejects preflight rather than silently starting a cold node.
- `navigation/localization/preflight_session_runtime.py`: one ROS context and
  independently serviced TF listener; fresh observation nodes per request.
  Configuration changes are rejected. ROS clock reset callbacks clear retained
  TF and invalidate a capture spanning the reset. A stopped TF executor rejects
  further admission. The service never publishes motion.
- `navigation/localization/tf_snapshot_barrier.py`: immediate zero-timeout lookup
  followed, only if necessary, by callback servicing and reacquisition until a
  0.5-second monotonic deadline. The completed stationary window's minimum stamp
  remains fixed. There is no five-callback limit and no fixed successful-path
  sleep. Lookup disappearance never returns an earlier saved transform.
  Running ROS callbacks cannot be preempted; a snapshot acquired after a
  callback overruns the deadline is rejected.
- `navigation/localization/ros_preflight.py`: injects the retained buffer and
  preserves existing frame, quaternion, finite-pose, freshness, future-time,
  stationary-window, and downstream certificate checks.
- `real_robot/autonomous_runner/runtime.py`: owns the session for the mission.

Standalone preflight commands retain their short-lived ROS lifecycle. The
snapshot deadline applies there as well. The follower's TF ownership and the
camera observer's timestamp-specific lookups are unchanged. Stationary sampling
and the configured observation window remain mandatory per admission; this
change does not reuse evidence from before the previous leg's motion.

## Diagnostics

The final `map -> odom` observation includes:

- `minimum_stamp_sec`
- `stationary_window_reacquisition_count`
- `stationary_window_wait_sec`
- `stationary_window_wait_timed_out`

The additional `preflight TF session` observation includes the service PID,
capture index, buffer reuse flag, and clock epoch. Successive mission preflights
should show the same PID and increasing capture indices, with newly captured
stationary samples each time. These fields make it possible to distinguish
buffer catch-up time from the time spent collecting fresh stationary evidence.

## Verification and limitations

Offline tests cover immediate admission, recovery after more than five
callbacks, missing/disappearing transforms, deadline expiry, malformed/future
transforms, retained-buffer identity across fresh nodes, clock reset rejection,
configuration changes, executor failure, JSON transport, inherited service
access, and cleanup after mission failure.

Final regression selection: **586 passed, 504 subtests passed, 1 deselected**.
The deselected test is
`test_runtime_localization_recovery_uses_scoped_permit_without_parent_input` in
`test_autonomous_stand_exploration.py`. Its fixture omits
`odom_execution_certificate_sha256`; the same failure was reproduced by loading
the unchanged `HEAD` autonomous runner into the test process. No permit checks
were weakened to accept that fixture. All 23 changed/new Python files in the
workspace parse with Python 3.10 grammar, and `git diff --check` passes.

ROS-specific construction uses test doubles locally; this machine has no ROS2
runtime. No robot motion, deployment, or end-to-end speedup measurement was
performed. On the ROS host, verify the new session diagnostic and stationary
sample timestamps during stopped dry-runs before measuring successive legs.

The Humble API compatibility checks used the upstream
[rclpy clock implementation](https://github.com/ros2/rclpy/blob/humble/rclpy/rclpy/clock.py)
and [tf2 buffer implementation](https://github.com/ros2/geometry2/blob/humble/tf2_ros_py/tf2_ros/buffer.py).
Humble's `JumpThreshold` requires both duration keywords; its Python buffer does
not itself install the reset callback used by this session owner.
