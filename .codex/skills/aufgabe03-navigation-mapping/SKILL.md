---
name: aufgabe03-navigation-mapping
description: >
  Handle Aufgabe 03 navigation and mapping work in the mii-amr repository:
  SLAM/map creation, saved map/Nav2/AMCL setup, offline A* path planning,
  waypoint following, LiDAR obstacle overlays, active replanning, collision
  avoidance, RViz visualization, experiment logs, and sensor/map failure
  analysis.
---

# Aufgabe 03 Navigation And Mapping

## Purpose

Use this skill for Aufgabe 03 work from `docs/tasks/Aufgabe_03.pdf`: mapping
the test room, navigating on the saved map, implementing path planning,
detecting temporary LiDAR obstacles, updating a run-local map, replanning while
driving, validating collision avoidance, and explaining sensor or mapping
limitations.

For report-ready current evidence, read:

```text
.codex/skills/aufgabe03-navigation-mapping/references/current-findings.md
```

## Task Sheet Scope

Aufgabe 03 asks for:

- use robot SLAM to create a test-room map and navigate in it
- implement and test a path-finding algorithm on that map
- add an obstacle from raw LiDAR data as an extension of the stationary map
- add an active component that updates the map while driving
- execute navigation on the planned path with active collision avoidance
- validate with a temporary obstacle
- intentionally find mapping failure modes and explain sensor/map limits

## Code Map

- `maps/aufgabe03/`: saved ROS map YAML/PGM files.
- `docs/setups/nav2_waypoint.txt`: lab runbook for Nav2, AMCL, RViz, two-stage
  waypoint following, and LiDAR obstacle replanning.
- `scripts/aufgabe03/map_path_planner.py`: offline stdlib A* planner for ROS
  trinary occupancy maps; writes dense path CSV, simplified waypoint CSV, and
  PPM visualization.
- `scripts/aufgabe03/arena_coverage_drive.py`: conservative mapping coverage
  motion; requires bringup and SLAM/Cartographer to already be running.
- `scripts/aufgabe03/arena_active_spin.py` and
  `scripts/aufgabe03/arena_geometry_localizer.py`: active spin localization and
  arena-derived pose prior.
- `scripts/aufgabe03/two_stage_waypoint_run.py`: public entrypoint for
  arena-prior localization, Nav2 staging, and custom waypoint following.
- `scripts/aufgabe03/follow_planned_waypoints.py`: custom TF/AMCL waypoint
  follower with scan hard-stop, RViz route markers, and optional run-local
  LiDAR obstacle replanning.
- `scripts/aufgabe03/lidar_obstacle_map.py`: ROS-free temporary obstacle
  overlay and A* replan artifact generation; does not modify the saved map.
- `scripts/aufgabe03/lidar_obstacle_debug_viz.py`: live RViz markers for scan
  points, obstacle ROI, accepted cells, inflated cells, and rejected points.
- `scripts/aufgabe03/arena_active_temporary_map_debug_viz.py`: read-only RViz
  publisher for the arena-active observed temporary map, inflated planning map,
  active-explore A* path, and candidate markers.
- `scripts/aufgabe03/analyze_waypoint_follow_runs.py`: summarizes waypoint and
  replan CSV logs for report evidence.

## Operating Rules

Start read-only:

```bash
pwd
git status --short
rg --files scripts/aufgabe03 maps/aufgabe03 results/aufgabe03 docs/setups
```

Before suggesting real robot motion, state the physical safety requirements:
clear arena, operator beside the robot, Ctrl+C ready, physical stop possible,
and a separate `/cmd_vel` stop terminal available.

Never execute physical TurtleBot motion automatically.

## Environment

Use `[workstation/container]` for ROS2, TurtleBot bringup, Cartographer, Nav2,
AMCL, `/scan`, `/odom`, `/cmd_vel`, and real robot motion.

Use `[MacBook]` for editing, Git, offline A* planning, CSV inspection, plotting,
and non-ROS report work.

For Aufgabe 03 real robot commands, prefer:

```bash
source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02
```

## Key Files

Current saved map:

```text
maps/aufgabe03/arena_1p898x3p9_auto.yaml
maps/aufgabe03/arena_1p898x3p9_auto.pgm
```

Current planned path artifacts:

```text
results/aufgabe03/aufgabe03_planned_path.csv
results/aufgabe03/aufgabe03_waypoints.csv
results/aufgabe03/aufgabe03_planned_path.ppm
```

Current run logs:

```text
results/aufgabe03/arena_coverage_runs.csv
results/aufgabe03/aufgabe03_waypoint_follow_runs.csv
results/aufgabe03/aufgabe03_arena_prior_two_stage_runs.csv
```

The last file may not exist until a two-stage run is logged.

## Preflight Checks

Before Nav2 or follower motion:

```bash
ros2 topic echo --once /scan
ros2 topic echo --once /odom
ros2 topic echo --once /amcl_pose
ros2 topic list | grep -E '^/(scan|odom|amcl_pose|initialpose)$'
ros2 topic info /initialpose --verbose
ros2 action list | grep navigate_to_pose
ros2 topic info /cmd_vel --verbose
```

Interpretation:

- `/scan` missing blocks mapping coverage, LiDAR obstacle updates, and safety.
- `/amcl_pose` missing blocks waypoint startup and two-stage validation.
- no `/initialpose` subscriber means the arena prior cannot seed AMCL.
- competing `/cmd_vel` publishers are unsafe unless the runbook explicitly
  allows known Nav2 publishers and no active Nav2 goal is running.

## Planning Workflow

Use the offline A* planner for path-finding evidence:

```bash
python3 scripts/aufgabe03/map_path_planner.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --start 0.005 -0.015 \
  --goal 1.005 0.285
```

Report:

- map file, resolution, origin, and occupancy assumptions
- requested vs snapped start/goal if snapping occurs
- dense path length and simplified waypoint count
- inflation radius and snap radius
- output CSV/PPM artifacts

## Waypoint Execution

Use `docs/setups/nav2_waypoint.txt` as the current command source. The intended
flow is:

```text
bringup -> Nav2 with saved map -> arena spin localization -> publish AMCL prior -> Nav2 to waypoint 0 -> custom follower
```

Typical dry run:

```bash
python3 scripts/aufgabe03/two_stage_waypoint_run.py \
  --waypoints results/aufgabe03/aufgabe03_waypoints.csv \
  --run-id arena_prior_two_stage_dry_001 \
  --dry-run
```

Typical real run uses the runbook's arena-specific flags. For the current
arena orientation with `--heater-wall-side=+x`, force the positive short wall as
the heater side:

- `--heater-wall-side=+x`
- `--arena-force-short-wall-side axis_positive`
- `--arena-force-short-wall-type heater`
- `--arena-active-allow-extra-cmd-vel-publishers`
- relaxed AMCL stability only when justified by collected logs

If `--arena-force-short-wall-type clean` is paired with
`--heater-wall-side=+x` and `axis_positive`, the arena prior can select the
mirrored branch after publishing `/initialpose`.

Successful follower evidence should show `status=completed` and
`reached_count=3` in `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`.

## RViz Debug Visualization

Use the runbook as the command source. Important live topics:

- `Path: /mii_amr/planned_path`
- `MarkerArray: /mii_amr/planned_waypoints`
- `MarkerArray: /mii_amr/run_local_obstacles`
- `Map: /mii_amr/arena_active/temporary_map`
- `Map: /mii_amr/arena_active/temporary_map_planning`
- `Path: /mii_amr/arena_active/explore_path`
- `MarkerArray: /mii_amr/arena_active/explore_candidates`

The arena-active observed map is free/occupied scan evidence without planner
inflation. The planning map is the hard-inflated grid used by active-explore
A*. The active-explore path is the short odom-frame curve currently selected
for localization recovery, not the full offline CSV route.

If these arena-active maps do not render while RViz fixed frame is `map`, switch
fixed frame to `odom` until AMCL provides a usable `map -> odom` transform.
The standalone helper publishes the same topics as the integrated runner, but
stop it before real recovery if duplicate publishers or timing load affect the
robot.

## LiDAR Obstacle And Replanning

The saved static map is not modified in place. Temporary obstacles are added to
a run-local overlay, inflated, and used for a replanned path. Validate in this
order:

1. Offline synthetic obstacle artifact generation.
2. Two-stage dry run with `--enable-lidar-map-replan`.
3. Robot artifact-only run with `--lidar-replan-artifact-only`.
4. Active temporary-obstacle run without artifact-only mode.

Useful offline command:

```bash
python3 scripts/aufgabe03/lidar_obstacle_map.py \
  --static-map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --synthetic-obstacle-base-x 0.45 \
  --synthetic-obstacle-base-y 0.0 \
  --robot-pose 1.0,1.0,0.0 \
  --goal-pose 1.005,0.285,0.0 \
  --run-id synthetic_lidar_overlay_test
```

For active runs, log and report:

- detected obstacle count
- candidate and filtered scan points
- raw/free/inflated obstacle cells
- old and new path lengths
- replan count and trigger reason
- generated run-local map, waypoint, path, and obstacle CSV artifacts
- whether the robot stopped, replanned, and resumed safely

For delayed obstacle validation where the robot should first follow the planned
route and only replan after a later scan blockage, use
`--run-local-map-initial-scan-mode none`. Otherwise an initial full-scan
run-local map can proactively replace/prune the CSV route before motion. The
latest successful late-obstacle setup used `--max-replans 4`,
`--run-local-map-update-mode full`, and `--follower-min-scan-range-m 0.30`.

## Analysis Workflow

Summarize waypoint evidence with:

```bash
python3 scripts/aufgabe03/analyze_waypoint_follow_runs.py
```

If an older root-level log is relevant:

```bash
python3 scripts/aufgabe03/analyze_waypoint_follow_runs.py \
  --input results/aufgabe03_waypoint_follow_runs.csv
```

When interpreting results, separate:

- planning artifacts from real robot execution
- Nav2 staging from custom follower behavior
- static map navigation from run-local obstacle replanning
- hard-stop collision avoidance from actual path replanning
- failed setup evidence from successful navigation evidence

## Common Failure Patterns

- `/scan` missing: wrong LiDAR model, bringup issue, topic not publishing, or
  ROS domain/network mismatch.
- `/amcl_pose` timeout: Nav2/AMCL not launched with the saved map, AMCL not
  receiving scans, or no pose prior.
- stale TF: map to base transform not updating fast enough; check TF, AMCL,
  CPU load, and stale thresholds before changing controller logic.
- waypoint timeout: speed/gain/tolerance combination too aggressive or the
  robot is not close enough to the planned path at handoff.
- mirrored arena prior: forced short-wall side/type does not match the physical
  heater/clean side; for `--heater-wall-side=+x` and `axis_positive`, use
  `--arena-force-short-wall-type heater`.
- automatic replan before driving: `--enable-lidar-map-replan` with an initial
  scan mode other than `none` performs startup run-local planning, not just
  obstacle-triggered replanning.
- persistent scan blockage after existing-map repair: replanning worked, but
  the live scan corridor remained blocked after the replan budget was exhausted.
- RViz arena-active publisher side effects: integrated map/path/candidate
  publishing can add DDS/CPU timing load on the robot; for real motion, keep
  `--no-arena-active-temporary-map-viz` and
  `--no-arena-active-explore-path-viz` if that was required for stable driving.
- unrecognized CSV schema: move or migrate old generated logs before appending.
- no obstacle artifact in completed run: likely pre-replan schema or replan not
  enabled; do not claim active map-update evidence from that row.

## Report Guidance

For Aufgabe 03 reports, use this structure:

1. Mapping setup: SLAM stack, coverage motion, saved map, map resolution.
2. Path planning: A* on trinary occupancy grid, inflation, snapping, waypoints.
3. Localization and navigation: AMCL/Nav2 staging, arena prior, custom follower.
4. Dynamic obstacle handling: LiDAR ROI, filtering, run-local overlay, replan.
5. Validation: successful runs, failed setup attempts, scan clearances, artifacts.
6. Limitations: 2D LiDAR height plane, reflective/transparent/low obstacles,
   sparse geometry, symmetric room ambiguity, stale TF/AMCL, map alignment, and
   dependence on correct `LDS_MODEL` and ROS networking.
