---
name: mii-amr-experiments
description: >
  Project-specific guidance for the mii-amr autonomous mobile robotics repository.
  Use when working on ROS2 Humble, Gazebo, TurtleBot, camera/green-marker pose
  tracking, simulation-vs-real experiments, endpoint analysis, probabilistic
  final-pose modeling, Aufgabe 03 SLAM/Nav2/path planning/LiDAR obstacle
  replanning, Aufgabe 04 logistics/QR/station/fleet coordination, generated
  results, bag files, or project documentation.
---

# MII AMR Experiments

## Goal

Work inside the `mii-amr` repo for reproducible TurtleBot Sim2Real experiments:
simulate motion, run the same motion on the real robot, measure final pose with
the camera tracker, compare endpoints, model final-pose uncertainty, and run
Aufgabe 03 mapping/navigation experiments with saved maps, A* paths, AMCL/Nav2,
and LiDAR obstacle replanning. Also support Aufgabe 04 logistics work with QR
scanning, station routing, puck transport assumptions, and two-robot
coordination.

## Environments

The user works across:

- lab workstation/container: ROS2 Humble, Gazebo, TurtleBot networking, real robot runs, bag recording
- MacBook Air M1: editing, Git, reports, plotting, lightweight Python analysis, camera work if camera is connected there
- UTM Ubuntu on MacBook: Linux/ROS practice and lightweight tests; do not assume reliable Gazebo or robot hardware access

When commands depend on environment, label them:

- `[workstation/container]`
- `[MacBook]`
- `[UTM Ubuntu]`

Prefer the workstation/container for ROS2, Gazebo, TurtleBot, Apptainer, and real robot commands.
Prefer the MacBook for editing, Git, CSV analysis, plotting, and report work.
For camera-tracker work, infer or ask where the camera is physically connected.

## Repo Map

- `scripts/aufgabe02/`: scripted drives, motion primitives, endpoint models,
  plotting, and analysis helpers
- `scripts/aufgabe03/`: mapping coverage, A* planning, arena-prior two-stage
  waypoint runs, LiDAR obstacle overlays, run-local replanning, and analysis
- `scripts/aufgabe04/`: logistics modules for QR scanning, station maps,
  mission state, puck transport assumptions, fleet coordination, navigation
  adapters, strict preflight, and single-segment station-route execution
- `scripts/common/run_with_bundle.sh`: task-agnostic physical-run debug bundle
  wrapper; use it to capture raw diagnostics around explicitly supplied real
  run commands
- `vision_tracker/`: camera calibration, HSV tuning, green-marker tracking, pose estimation, start-pose gate
- `maps/aufgabe03/`: saved ROS map YAML/PGM files
- `docs/tasks/`: assignment PDFs
- `docs/setups/nav2_waypoint.txt`: current Aufgabe 03 lab runbook
- `docs/setups/aufgabe04_*.md`: draft Aufgabe 04 QR, logistics, real parkour,
  and two-robot runbooks
- `results/`: generated CSVs, plots, logs, analysis text
- `results/aufgabe04/`: QR scan, station visit, mission, fleet event, and real
  parkour evidence targets
- `results/real_runs/`: raw/debug bundles for physical runs; do not treat these
  as report evidence by themselves
- `bags/`: ROS bag outputs; do not commit
- `tests/`: focused unit tests for pure logic

## Current References

For Aufgabe 03 navigation and mapping workflows, prefer the dedicated skill:
`aufgabe03-navigation-mapping`.

For Aufgabe 04 logistics, QR scanning, station routing, puck transport, or
two-robot coordination, prefer the dedicated skill: `aufgabe04-logistics`.

For current experiment findings across Aufgabe 02 and Aufgabe 03, read:

```text
.codex/skills/aufgabe03-navigation-mapping/references/current-findings.md
```

## Operating Rules

Start read-only:

```bash
pwd
git status --short
rg --files -g '!bags/**'
```

Before editing, inspect relevant files. Do not revert unrelated user changes.
Prefer minimal targeted changes. Use `rg` and exact file reads over broad scans.
Verify current script behavior before claiming exact outputs.

## Command Style

Give exact commands and short checklists.
Say what must already be running or sourced.
Do not over-explain basic ROS concepts unless asked.

Run tests with:

```bash
python3 -m unittest discover tests
```

## Data And Git Rules

Generated experiment outputs go in `results/` unless requested otherwise.
Do not commit `bags/`, `*.db3`, `*.bag`, `*.bag.active`, or large robot logs.
CSV result files may be committed if they document experiments.

Before editing/deleting generated outputs, check tracking:

```bash
git ls-files results vision_tracker/data bags
git check-ignore -v bags/example.db3
```

Treat destructive cleanup, bag deletion, reset/clean, and history rewriting as unsafe. Flag before suggesting.

## Simulation Runs

Use for Gazebo/TurtleBot simulated motion and repeated simulated endpoint collection.

Before suggesting run commands, verify current script behavior.
Typical command:

```bash
./scripts/aufgabe02/run_experiment.sh 15
```

Optional metadata:

```bash
RUN_MODE=linear-forward RUN_SPEED=0.1 RUN_DISTANCE=30cm ./scripts/aufgabe02/run_experiment.sh 15
```

Flag before suggesting:

```bash
ros2 service call /reset_simulation std_srvs/srv/Empty
```

because it resets the active simulation.

## Real Robot Runs

Use only for physical TurtleBot, camera tracker, or real endpoint collection.

Never execute real robot motion automatically.
Any command that publishes `/cmd_vel` needs a safety note:

- clear test area
- operator nearby
- terminal ready for Ctrl+C
- physical stop possible

Camera setup:

```bash
python3 vision_tracker/list_cameras.py
python3 vision_tracker/tune_hsv.py
python3 vision_tracker/calibration.py
python3 vision_tracker/calibration.py --verify
python3 vision_tracker/main.py
```

Real run:

```bash
./scripts/aufgabe02/run_real_experiment.sh
```

Explicit run ID:

```bash
./scripts/aufgabe02/run_real_experiment.sh run_real_014
```

Before real runs:

- verify runtime path; script may assume `/workspace/mii-amr`
- run `vision_tracker/main.py` first so `results/aufgabe02/latest_tracker_pose.csv` is fresh
- verify start-pose gate behavior before motion
- generate next run ID if needed:

```bash
python3 scripts/common/next_real_run_id.py
```

For navigation/logistics real runs, wrap the actual run command so failed runs
also produce a debug bundle:

```bash
scripts/common/run_with_bundle.sh run_001 -- COMMAND ...
```

The bundle is passive evidence capture only. It does not replace strict
preflight, operator confirmation, or the safety-stop terminal.

## Aufgabe 03 Navigation And Mapping

Use for SLAM/map creation, saved-map navigation, A* planning, waypoint
following, active LiDAR obstacle handling, and collision-avoidance validation.

Current key files:

- `maps/aufgabe03/arena_1p898x3p9_auto.yaml`
- `results/aufgabe03/aufgabe03_planned_path.csv`
- `results/aufgabe03/aufgabe03_waypoints.csv`
- `results/aufgabe03/arena_coverage_runs.csv`
- `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`

Core commands:

```bash
python3 scripts/aufgabe03/map_path_planner.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --start 0.005 -0.015 \
  --goal 1.005 0.285
```

```bash
python3 scripts/aufgabe03/two_stage_waypoint_run.py \
  --waypoints results/aufgabe03/aufgabe03_waypoints.csv \
  --run-id arena_prior_two_stage_dry_001 \
  --dry-run
```

Before real motion, follow `docs/setups/nav2_waypoint.txt`, verify `/scan`,
`/odom`, `/amcl_pose`, `/initialpose`, `/navigate_to_pose`, and `/cmd_vel`
publishers, and require the real-robot safety checklist.

For obstacle evidence, distinguish:

- static A* path artifacts
- successful static waypoint following
- LiDAR obstacle overlay artifacts
- active run-local replan logs and artifacts

Do not claim active obstacle validation unless the run log contains replan
diagnostics and generated run-local map/path/waypoint artifacts.

## Aufgabe 04 Logistics

Use for QR scanning, station routing, puck transport assumptions, single-robot
mission state, and two-robot coordination.

Current code tree:

- `scripts/aufgabe04/qr_scanning/`: pure QR payload parsing, station order, and
  future onboard camera node
- `scripts/aufgabe04/stations/`: station poses, approach targets, and station
  visit routing
- `scripts/aufgabe04/logistics/`: mission state, puck transport assumptions, and
  mission CSV logging
- `scripts/aufgabe04/fleet/`: station locks, right-before-left policy, robot
  status, and conflict detection
- `scripts/aufgabe04/navigation/`: route adapter around Aufgabe 03 planning,
  waypoint generation, pure route/diagnostics gates, ROS runtime config,
  strict preflight, single-segment station route runner, simple waypoint
  follower, and segment run logging

Current single-segment dry run:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py \
  --dry-run \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

Current tests:

```bash
python3 -m unittest discover tests/aufgabe04
python3 -m unittest tests.test_run_with_bundle
```

For physical Aufgabe 04 runs, follow
`docs/setups/aufgabe04_real_parkour_checklist.md` and wrap the real command
with `scripts/common/run_with_bundle.sh`, passing matching namespace/topic/frame
options to the wrapper and the wrapped command. Keep most logic pure first. Add
ROS camera integration, multi-segment station-route motion, and two-robot CLIs
only after offline tests and safety runbooks are in place.

## Analysis

Use CSVs as source of truth, but verify current paths from scripts/results.

Common files:

- `results/aufgabe02/scripted_drive_runs.csv`: simulation endpoints
- `results/aufgabe02/real_scripted_drive_runs.csv`: real tracker/odometry endpoints
- `results/aufgabe02/real_start_pose_checks.csv`: accepted start-pose checks
- `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`: navigation/replan logs
- `results/aufgabe03/arena_coverage_runs.csv`: mapping coverage attempts

Common commands:

```bash
python3 scripts/aufgabe02/analyze_probabilistic_endpoint_model.py
python3 scripts/aufgabe02/build_motion_primitives_model.py
python3 scripts/aufgabe03/analyze_waypoint_follow_runs.py
```

Before relying on analysis scripts, inspect hard-coded run filters.

For endpoint analysis, compute:

- valid run count
- start-to-final displacement
- simulated endpoint mean/std
- real tracker endpoint mean/std
- dx, dy, Euclidean endpoint error
- yaw error
- covariance if useful
- outliers/residuals

Always report units:

- x, y, dx, dy, endpoint error: meters
- yaw/yaw error: degrees

## Probabilistic Model

Prefer simple reproducible statistics:

- empirical mean endpoint
- empirical covariance over `(x, y)`
- optional yaw mean/std separately
- confidence ellipse or sampled endpoints if useful

Do not overclaim with small sample sizes.

## Coding Guidance

Preserve shared CSV schemas unless explicitly changing them.
Keep camera tuning centralized in `vision_tracker/config.py`.
Keep start-pose behavior consistent across:

- `vision_tracker/main.py`
- `vision_tracker/start_pose.py`
- `vision_tracker/start_pose_gate.py`

Add focused unit tests for pure logic.
Avoid tests requiring camera, ROS graph, Gazebo, or robot unless explicitly requested.

## Documentation Guidance

For reports or presentations:

- separate method, result, and interpretation
- distinguish odometry pose from camera-tracker pose
- include run count and units
- state limitations: homography, marker detection, start-pose tolerance, odometry drift, surface/battery differences, limited sample size
- keep wording technical and concise
