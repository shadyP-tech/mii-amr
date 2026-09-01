# Aufgabe 04 Real Parkour Checklist

## Migration Status

End-to-end Aufgabe04 hardware migration is currently **blocked**. This checklist
is necessary but not sufficient; complete the staged gate in
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md) first. The
passive synchronized real viewpoint adapter and sealed unloaded-segment wrapper
are implemented, but have not yet been validated on the robot. Strict onboard
QR-event production, mission/follower integration, an independent ROS command
guard/mux, automatic execution-evidence manifests, measured carrier/loaded
dynamics, and the active fleet runtime are still not implemented. Follow
[`aufgabe04_real_pipeline.md`](aufgabe04_real_pipeline.md) for passive survey
and profile preparation.

Until those blockers are closed, limit new real-world work to passive sensor and
calibration capture or separately authorized, unloaded single-segment
validation. Do not present either as a completed logistics mission.

Before any real robot motion:

- Clear the arena and station approach zones.
- Keep an operator beside each robot.
- Keep Ctrl+C in the active runner terminal and physical stop access ready.
- Keep a separate terminal ready to publish zero `Twist` to the exact resolved
  `/cmd_vel` topic.
- Verify `/scan`, `/odom`, TF, configured AMCL/SLAM data, map, namespace, and
  `/cmd_vel` ownership.
- Verify exactly one localization source owns `map -> odom`: AMCL or SLAM
  Toolbox, not both.
- Verify no active Nav2 goal, competing controller, or custom follower is
  publishing velocity commands.
- For custom follower motion, the resolved `/cmd_vel` topic must be owned by
  the follower only during runtime.
- Start with dry-run and single-robot checks before two-robot operation.

## Sealed Artifact and Route Gate

Before an unloaded single-segment motion trial, verify all applicable artifacts
belong to the same run:

- frozen map bundle matches the exact YAML, referenced image, semantic map ID,
  and planning frame loaded for localization/planning;
- candidate snapshot and identity registry match the surveyed catalog;
- a logistics route matches the immutable task snapshot's exact station order
  and its parent survey manifest;
- route CSV and diagnostics match the execution route certificate, including
  frame, route kind, waypoint count, tube radius, exact-vertex policy, command
  owner, map bundle, and candidate snapshot;
- robot namespace and runtime command-owner identity match the certificate;
- controller profile, footprint, localization uncertainty, and clearance margin
  are the measured values intended for this run.
- for uncertainty-aware odom execution, the branch-proof ID identifies a known
  physical start or asymmetric landmark, the frozen `map <- odom` certificate
  matches the route, and every segment retains a strictly positive uncertainty
  margin; five stable AMCL means alone do not establish absolute accuracy.

For loaded motion, additionally require a measured carrier profile, confirmed
retention, current fenced puck custody, loaded stopping-distance evidence, and
the loaded footprint in planning and fleet collision checks. These are not yet
wired into the ROS runner, so loaded mission motion remains blocked.

The ROS-free `cmd_vel_guard.py` state machine is not an active watchdog. Do not
credit it as crash-independent velocity authority until a deployed guard/mux
has been tested to output zero velocity on owner/lease loss, replay, stale or
future commands, clock rollback, and process termination.

## Single-Segment Bringup Order

1. Start TurtleBot3 bringup.
2. Start saved-map localization or another explicit `map -> base_footprint`
   localization source.
3. Set the initial pose in RViz and verify the LiDAR overlay aligns with the map.
   If RViz reports `base_scan` message-filter queue overflow, treat it as a TF
   or visualization backlog symptom, not as the controller's stop cause. Verify
   `base_footprint <- base_scan` with
   `ros2 run tf2_ros tf2_echo base_footprint base_scan`, and do not continue if
   the static edge is absent or changes.
4. Confirm there is no active Nav2 goal before handing off to the custom
   Aufgabe 04 follower.
5. Do not feed the old `station_route.csv` artifacts to the segment runner.
   They lack the sealed route kind, certificate, and mission root and are now
   intentionally rejected. Complete the simulation-to-real gate and use the
   full task-route inputs documented in `aufgabe04_arrival_pose_survey.md`.

6. If the dry run passes, keep a separate stop terminal ready for the resolved
   velocity topic printed by the dry run. For example:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'
```

Use `/robot1/cmd_vel` instead when the dry-run output resolves the robot's
velocity topic under a namespace.

7. The dedicated `real_robot/entrypoints/run_unloaded_segment.py` adapter retains every
   sealed mission-chain input and is dry-run by default. Its `--execute` path is
   only for a separately authorized, unloaded single-leg validation after the
   staged gate and physical precautions pass. It does not clear real logistics
   mission execution.

Every physical mission requires an operator to type `RUN` before its first
motion. A standalone `run_single_station_segment.py` invocation still has no
`--yes` bypass and remains interactive. The autonomous exploration wrapper may
skip repeated child prompts only for the exact routine legs covered by its
immutable, one-use `mission_leg_motion_permit`; each child still repeats the
dry-run/live gates, validates the sealed artifacts, and consumes its receipt
immediately before entering the follower. The initial mission `RUN` may also
cover a bounded, same-leg, same-target pre-motion recovery after either a
certified-start mismatch or an exact zero-motion `map <- odom` consistency
stop. That recovery is valid only after fresh stationary localization, a new
sealed route/certificate, a passed dry run, and an exact one-use
`startup_reseal_motion_permit`; generic, malformed, target-changing, budget-
exhausted, and post-motion failures remain terminal. No additional `RUN` is
requested for an admitted recovery. In semantic logs, `motion_started` marks
the child execution-attempt handoff before the follower runs; use the terminal
`motion_published` field and controller trace to determine whether a nonzero
Twist was actually published. Keep an operator beside the robot throughout the
mission.

The optional exact-two startup active-localization phase is separate from that
mission authority. It can appear only after a successful stopped AMCL
admission and a typed initial route-uncertainty rejection. Clear the robot's
full swept rotation envelope, then type `LOCALIZE` only if you intend to permit
that one bounded, translation-free in-place rotation. A completed rotation
must end with repeated zero commands and stopped-odom proof, followed by a new
stationary AMCL/TF admission and a new initial plan. It does not authorize the
plan: review the subsequent readiness output and type the separate `RUN` only
after all normal route gates pass. Wrong confirmation, another `/cmd_vel`
publisher, stale scan/odom, an obstacle, excessive translation, missing trace,
or an exhausted attempt budget leaves the robot stopped.

For the next operator-authorized trial, disable nonessential RViz LaserScan,
Path, and long-history displays. Start an observe-only external capture of
`/tf` and `/tf_static` before motion and preserve it with the bundle so the
failure interval can be split into `map -> odom` and
`odom -> base_footprint`. The bundle's one-shot pre/post TF probes cannot prove
which edge paused during the run. This capture must never publish `/cmd_vel`.

Every physical run should also be wrapped in a real-run debug bundle. The
bundle records evidence only; it does not replace the strict dry-run/preflight
inside `run_single_station_segment.py` and it never publishes motion by itself.
Keep the bundle topic options matched to the wrapped command topic options.

For the detected-station route artifacts, use the dedicated wrapper after the
dry run has passed. The wrapper first collects bundle diagnostics, then the
inner runner pauses and explicitly prompts for `2D Pose Estimate` immediately
before ROS preflight. Click the pose estimate at that prompt, press Enter, and
type `RUN` only after the inner preflight passes:

```bash
scripts/aufgabe04/navigation/entrypoints/run_first_detected_station_segment_with_bundle.sh \
  aufgabe04_first_detected_real_001 \
  --allow-idle-nav2-publishers \
  --operator-note "first detected station pre-approach"
```

Do not reconstruct the removed abbreviated namespaced example by adding only
topic flags. A future real command must retain every sealed mission-chain input
from the simulation admission command and add matching namespace/topic/frame
settings to both the evidence wrapper and runner.

After the run, create the upload/debug archive printed in
`results/real_runs/run_001/archive_hint.txt`. The same wrapper can be used for
Aufgabe 03 real runs by wrapping the appropriate Aufgabe 03 command.

## Scan Safety Evidence

Before using a `safety_stop` as evidence that the robot was physically too
close to an obstacle, inspect the semantic JSONL details for the stop. A valid
`obstacle too close` claim must include a `nearest_valid_range_m` below the
configured threshold and the reported `/scan` `range_min_m`/`range_max_m`.
That one range is sufficient to stop immediately, but it is not sufficient to
modify the route. A transient keepout additionally requires the structured
`stationary_obstacle_confirmation` evidence: at least three fresh, distinct
post-stop samples, a bounded front-range/map-hit cluster, stationary map and
odom poses, and bounded map/odom offset and yaw-offset spread. Inspect the
thresholds and measured spreads rather than treating `confirmed=true` alone as
the claim.

Invalid LiDAR samples are evidence too, but they are a different claim:

- `no valid scan ranges` means the fresh scan had no globally valid ranges
  after applying `range_min`/`range_max`; treat it as unsafe sensor data, not
  as physical clearance.
- `no valid front-sector scan ranges` means forward clearance was unknown in
  the control sector; the follower must clamp forward velocity rather than
  treat the front as clear.
- Rejected below-min, above-max, and non-finite sample counts must be recorded
  before concluding whether a near-zero reading was a real obstacle or a scan
  artifact.
- `clearance-limited motion floor` means obstacle scaling would have requested
  nonzero motion below the configured physical floor. The emitted command must
  be exactly zero while the same stationary confirmation gate decides between
  a bounded replan, a separately confirmed clear front, or a fail-closed stop.

For real TurtleBot3 runs, record live `/scan` diagnostics confirming
`range_min`, `range_max`, invalid sample counts, and front-sector clamp behavior
before making real-robot safety claims. Offline unit tests only prove the pure
filtering logic; workstation/ROS validation must confirm that `safety_stop`
JSONL carries the structured details while the CSV `stop_reason` remains
compact.

For bundled physical motion, also inspect `controller_trace.jsonl`. It records
the map/odom poses, route revision, target and pursuit indices, nominal and
effective commands, front-clearance summary, active certified segment, and
route-tube distances for motion cycles and critical zero/stop transitions. A
missing trace on an authorized bundled run is missing evidence, not permission
to infer controller behavior from the compact CSV.

## Preflight Gate

The preflight must pass before follower motion starts:

- resolved `/cmd_vel` must match the intended robot, for example `/cmd_vel` vs
  `/robot1/cmd_vel`
- `/scan` and `/odom` must be fresh by header timestamp and local receipt time
- TF must provide fresh `map -> base_footprint` and `odom -> base_footprint`
- AMCL must be fresh when AMCL is the localization source; SLAM Toolbox must be
  the only `map -> odom` source when SLAM is the localization source
- real-robot runs require `use_sim_time=false` unless intentionally overridden
- `/cmd_vel` ownership fails closed when another controller might own velocity;
  runtime custom follower motion stops if any external publisher appears on the
  resolved velocity topic
- Nav2 publisher count alone is not enough; check active NavigateToPose or
  controller state where possible
- the preflight AMCL no-motion request retains its own service and readiness
  timeout; the separate runtime request defaults to the namespace-relative
  `request_nomotion_update` service and a fail-closed budget of 2.0 s or less

The runner writes semantic events such as `run_started`, `runtime_resolved`,
`preflight_passed` or `preflight_failed`, `motion_started`, `safety_stop`, and
`run_finished` to the configured JSONL semantic log. The CSV result row records
the semantic log and preflight JSON paths. The bundle remains external evidence
capture only: terminal output, command, environment/git/ROS diagnostics, and
pre/post topic state.

A successful dry run, parser/unit tests, and a no-motion preflight show only
that the gates and configuration are wired correctly. They do not physically
validate AMCL refresh recovery, certified-route continuation, obstacle
detection, or adaptive replanning.

Zero-length legs are no-op evidence only. They must be run with `--allow-noop`;
the runner logs `motion_published=false` and does not publish motion.
