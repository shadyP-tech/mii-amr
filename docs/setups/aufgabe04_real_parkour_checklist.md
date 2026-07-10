# Aufgabe 04 Real Parkour Checklist

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

## Single-Segment Bringup Order

1. Start TurtleBot3 bringup.
2. Start saved-map localization or another explicit `map -> base_footprint`
   localization source.
3. Set the initial pose in RViz and verify the LiDAR overlay aligns with the map.
4. Confirm there is no active Nav2 goal before handing off to the custom
   Aufgabe 04 follower.
5. Run the station-segment dry run. Inspect the resolved topics before motion:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --dry-run \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json \
  --semantic-log results/aufgabe04/run_events/dry_run_001.jsonl
```

6. If the dry run passes, keep a separate stop terminal ready for the resolved
   velocity topic printed by the dry run. For example:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'
```

Use `/robot1/cmd_vel` instead when the dry-run output resolves the robot's
velocity topic under a namespace.

7. Run the real segment only after the dry-run gates are clean:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json \
  --preflight-json results/real_runs/run_001/aufgabe04_preflight.json \
  --semantic-log results/real_runs/run_001/aufgabe04_events.jsonl
```

The first physical validation requires typing `RUN` before motion. Do not use
`--yes` for the first real run; reserve it for repeated runs only after the
same dry-run/preflight sequence has already been validated and an operator
remains beside the robot.

Every physical run should also be wrapped in a real-run debug bundle. The
bundle records evidence only; it does not replace the strict dry-run/preflight
inside `run_single_station_segment.py` and it never publishes motion by itself.
Keep the bundle topic options matched to the wrapped command topic options.

Example namespaced real run:

```bash
scripts/common/run_with_bundle.sh \
  --namespace robot1 \
  --cmd-vel-topic cmd_vel \
  --scan-topic scan \
  --odom-topic odom \
  --amcl-topic amcl_pose \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_footprint \
  run_001 \
  -- \
  python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
    --namespace robot1 \
    --cmd-vel-topic cmd_vel \
    --scan-topic scan \
    --odom-topic odom \
    --amcl-topic amcl_pose \
    --map-frame map \
    --odom-frame odom \
    --base-frame base_footprint \
    --leg-index 1 \
    --route-csv results/aufgabe04/routes/station_route.csv \
    --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json \
    --preflight-json results/real_runs/run_001/aufgabe04_preflight.json \
    --semantic-log results/real_runs/run_001/aufgabe04_events.jsonl
```

After the run, create the upload/debug archive printed in
`results/real_runs/run_001/archive_hint.txt`. The same wrapper can be used for
Aufgabe 03 real runs by wrapping the appropriate Aufgabe 03 command.

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

The runner writes semantic events such as `run_started`, `runtime_resolved`,
`preflight_passed` or `preflight_failed`, `motion_started`, `safety_stop`, and
`run_finished` to the configured JSONL semantic log. The CSV result row records
the semantic log and preflight JSON paths. The bundle remains external evidence
capture only: terminal output, command, environment/git/ROS diagnostics, and
pre/post topic state.

Zero-length legs are no-op evidence only. They must be run with `--allow-noop`;
the runner logs `motion_published=false` and does not publish motion.
