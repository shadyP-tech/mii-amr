---
name: experiment-run-protocol
description: >
  Prepare reproducible AMR experiment run protocols for the mii-amr project
  before simulation scripted drives, real TurtleBot scripted drives,
  camera-tracker pose measurements, Sim2Real paired runs, or Aufgabe 03
  SLAM/Nav2/waypoint/LiDAR obstacle runs. Use when the user asks to run, repeat,
  record, validate, or compare experiment data collection.
---

# Experiment Run Protocol

## Purpose

Create reproducible run protocols for `mii-amr` experiments. Prioritize exact
run order, terminal ownership, safety, generated files, CSV fields, and
post-run validation. The main project risk is inconsistent data collection.

## Project Assumptions

- Repo: `mii-amr`.
- ROS2/Gazebo/TurtleBot work normally happens on the workstation/container.
- Camera tracking may happen on a separate camera host, often the MacBook.
- If the camera host differs from the run host, `results/aufgabe02/latest_tracker_pose.csv`
  must be visible at the same repo path read by
  `scripts/aufgabe02/run_real_experiment.sh`.
- MacBook camera host path:
  `/Users/stephpark/Documents/stephsWorld/mii-amr`.
- MacBook tracker environment: `conda activate mii-amr`.
- `scripts/aufgabe02/run_real_experiment.sh` currently expects
  `/workspace/mii-amr`.
- `RUN_MODE`, `RUN_SPEED`, and `RUN_DISTANCE` in `run_experiment.sh` are run-ID
  metadata. Current motion is controlled by constants in
  `scripts/aufgabe02/scripted_drive.py` and
  `scripts/aufgabe02/real_scripted_drive.py` unless the code has changed.
- Aufgabe 03 real robot navigation uses the lab/container ROS environment and
  typically requires `LDS_MODEL=LDS-02`, `/scan`, `/odom`, Nav2/AMCL, and a
  saved map under `maps/aufgabe03/`.

## Known Project Scripts

- `scripts/aufgabe02/run_experiment.sh`
- `scripts/aufgabe02/run_real_experiment.sh`
- `scripts/aufgabe02/scripted_drive.py`
- `scripts/aufgabe02/real_scripted_drive.py`
- `scripts/common/next_real_run_id.py`
- `scripts/common/run_with_bundle.sh`
- `scripts/aufgabe03/arena_coverage_drive.py`
- `scripts/aufgabe03/map_path_planner.py`
- `scripts/aufgabe03/two_stage_waypoint_run.py`
- `scripts/aufgabe03/follow_planned_waypoints.py`
- `scripts/aufgabe03/lidar_obstacle_map.py`
- `scripts/aufgabe03/analyze_waypoint_follow_runs.py`
- `scripts/aufgabe04/navigation/run_single_station_segment.py`
- `scripts/aufgabe04/navigation/ros_preflight.py`
- `docs/setups/nav2_waypoint.txt`
- `docs/setups/aufgabe04_real_parkour_checklist.md`
- `vision_tracker/main.py`
- `vision_tracker/start_pose_gate.py`
- `vision_tracker/config.py`

## Response Contract

For full protocol requests, produce:

1. Pre-run checklist
2. Commands in order
3. Files that should be created
4. Data columns to check
5. Post-run validation
6. Likely failure cases

For short questions, answer directly and include only the relevant protocol
risks.

Always label command terminals, for example:

- `Terminal A [workstation/container: simulation]`
- `Terminal B [camera host: tracker]`
- `Terminal C [workstation/container: real robot]`
- `Terminal S [workstation/container: safety stop]`

If the experiment type is unclear, ask one concise question choosing among:

- simulation scripted drive
- real TurtleBot scripted drive
- camera tracker final pose measurement
- Sim2Real paired run
- Aufgabe 03 mapping/navigation/obstacle run

## Safety And Data Rules

- Do not execute or imply automatic execution of physical TurtleBot motion.
- For real robot motion, require a clear test area, operator nearby, Ctrl+C
  ready, and physical stop option.
- Always include a safety-stop command for real runs:
  `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}"`
- For physical navigation/logistics runs, wrap the real command with
  `scripts/common/run_with_bundle.sh RUN_ID -- COMMAND ...` so failed runs also
  produce a debug bundle under `results/real_runs/<run_id>/`.
- The run bundle is evidence capture only. It does not replace the wrapped
  command's strict preflight, typed operator confirmation, or physical stop
  readiness.
- When using namespaces, pass matching `--namespace`, topic, and frame options
  to both the bundle wrapper and the wrapped command so the bundle records the
  intended robot's `/cmd_vel`, `/scan`, `/odom`, AMCL, and TF evidence.
- Always include a fresh tracker-pose check before real bag recording.
- Do not suggest committing generated bags or temporary data.
- Never suggest staging `bags/`, `*.db3`, `*.bag`, `*.bag.active`,
  `results/aufgabe02/*_bag.log`, `results/aufgabe02/latest_tracker_pose.csv`, or
  `vision_tracker/data/*`.
- CSV result files may be committed only when they are deliberate documented
  experiment results.

## Common Checks To Include When Relevant

- `pwd`
- `git status --short`
- `scripts/common/run_with_bundle.sh --help`
- `source /opt/ros/humble/setup.bash`
- `ros2 topic list | grep -E '^/(cmd_vel|odom)$'`
- `ros2 topic echo /odom --once`
- Real TurtleBot env:
  `source /opt/tb3_src_ws/install/setup.bash`,
  `export ROS_DOMAIN_ID=30`,
  `export ROS_LOCALHOST_ONLY=0`,
  `export TURTLEBOT3_MODEL=burger`,
  `export LDS_MODEL=LDS-01`
- Aufgabe 03 real navigation env usually uses:
  `export LDS_MODEL=LDS-02`
- Camera checks:
  `python3 vision_tracker/list_cameras.py`,
  `python3 vision_tracker/calibration.py --verify`,
  `python3 vision_tracker/main.py`
- Fresh latest-pose check:
  verify `results/aufgabe02/latest_tracker_pose.csv` exists, is recent, has
  `valid_pose` true, enough detected markers, and finite `x`/`y`/`yaw` values.

## Protocol Rules

### Simulation Scripted Drive

- State that Gazebo/TurtleBot simulation must already be running.
- Use `scripts/aufgabe02/run_experiment.sh` with the requested run count.
- Mention that it creates `bags/` and `results/`, records `/cmd_vel` and
  `/odom`, and may record `/imu` and `/battery_state` when present.
- Expected files:
  - `bags/simulation_run_<mode>_<speed>_<distance>_<NN>/`
  - `results/aufgabe02/<run_id>_bag.log`
  - `results/aufgabe02/scripted_drive_runs.csv`
- Check simulation CSV columns:
  `timestamp`, `run_id`, `odom_start_x`, `odom_start_y`,
  `odom_start_yaw_deg`, `odom_final_x`, `odom_final_y`,
  `odom_final_yaw_deg`, `notes`.

### Real TurtleBot Scripted Drive

- Prefer one physical run ID at a time unless the user explicitly asks for
  batch real runs.
- Split commands into tracker, run, and safety terminals.
- Generate or request a concrete `run_id`, but do not hard-code stale examples.
- Require fresh tracker pose before running.
- State that `scripts/aufgabe02/run_real_experiment.sh` runs
  `vision_tracker/start_pose_gate.py` before bag recording and prompts before
  motion.
- Expected files:
  - `bags/real/<run_id>/`
  - `results/aufgabe02/<run_id>_bag.log`
  - `results/aufgabe02/real_scripted_drive_runs.csv`
  - `results/aufgabe02/real_start_pose_checks.csv`
  - `results/aufgabe02/latest_tracker_pose.csv`
  - `vision_tracker/data/vision_<timestamp>.csv`
- Check real CSV columns:
  `timestamp`, `run_id`, `tracker_start_x`, `tracker_start_y`,
  `tracker_start_yaw_deg`, `tracker_final_x`, `tracker_final_y`,
  `tracker_final_yaw_deg`, `odom_start_x`, `odom_start_y`,
  `odom_start_yaw_deg`, `odom_final_x`, `odom_final_y`,
  `odom_final_yaw_deg`, `notes`.
- Check start-pose columns:
  `run_id`, `measured_x`, `measured_y`, `measured_yaw_deg`, `dx`, `dy`,
  `position_error_m`, `yaw_error_deg`, `pose_age_sec`, `stable_time_sec`,
  `accepted`.

### Camera Tracker Final Pose Measurement

- Use when only measuring pose from the tracker.
- Include camera listing, calibration verify, and tracker startup.
- Tell the user to keep the robot stationary long enough for several valid
  rows.
- Expected files:
  - `vision_tracker/data/vision_<timestamp>.csv`
  - `results/aufgabe02/latest_tracker_pose.csv`
- Check `valid_pose`, `num_detected`, `x`/`y`/`yaw`, and marker coordinates.

### Sim2Real Paired Run

- Keep the motion definition fixed between simulation and real.
- Current scripted motion is `0.1 m/s` forward for `3.0 s` unless code changed.
- Run simulation first, then one real protocol per physical run ID.
- Do not rename bag files to encode mappings.
- Compare simulation odom final pose, real tracker final pose, real odom final
  pose, start-pose consistency, `dx`, `dy`, Euclidean endpoint error, yaw error,
  run count, and outliers.
- For a rough probabilistic endpoint model, use empirical endpoint mean/std and
  covariance over valid final poses; do not overclaim with small run counts.

### Aufgabe 03 Mapping / Navigation / Obstacle Run

- Use when collecting evidence for SLAM map creation, saved-map navigation,
  A* path following, LiDAR obstacle overlays, active replanning, or collision
  avoidance.
- Start from `docs/setups/nav2_waypoint.txt`; it is the current lab runbook.
- Use four terminal roles:
  - `Terminal A [workstation/container: robot bringup]`
  - `Terminal B [workstation/container: Nav2 with saved map]`
  - `Terminal C [workstation/container: preflight/run]`
  - `Terminal S [workstation/container: safety stop]`
- Required preflight topics/actions:
  `/scan`, `/odom`, `/amcl_pose`, `/initialpose`, `/cmd_vel`,
  `/navigate_to_pose`.
- Check `/initialpose` has a subscriber before relying on the arena prior.
- Check `/cmd_vel` publishers and ensure no active Nav2 goal competes with the
  custom follower handoff.
- For the current arena prior with `--heater-wall-side=+x` and forced
  `axis_positive`, use `--arena-force-short-wall-type heater`; using `clean`
  can mirror the localization branch.
- Run offline planning or dry-run validation before motion:

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

- For real two-stage runs, require explicit operator confirmation and note that
  the script prompts for `RUN` before moving.
- For LiDAR obstacle work, validate in this order:
  offline synthetic overlay, dry-run replan flags, robot artifact-only mode,
  then active temporary-obstacle validation.
- For delayed obstacle tests, use `--run-local-map-initial-scan-mode none` so
  the route starts from the CSV and only replans after a later scan blockage.
  Use `--run-local-map-update-mode full`, `--max-replans 4`, and
  `--follower-min-scan-range-m 0.30` when matching the latest successful
  late-obstacle protocol.
- If integrated arena-active RViz publishing affects robot timing, keep
  `--no-arena-active-temporary-map-viz` and
  `--no-arena-active-explore-path-viz` for real motion. Use the standalone
  read-only helper only for preview, then stop it before the real recovery to
  avoid duplicate publishers.
- Expected files:
  - `results/aufgabe03/aufgabe03_planned_path.csv`
  - `results/aufgabe03/aufgabe03_waypoints.csv`
  - `results/aufgabe03/aufgabe03_planned_path.ppm`
  - `results/aufgabe03/arena_coverage_runs.csv`
  - `results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`
  - `results/aufgabe03/aufgabe03_arena_prior_two_stage_runs.csv`
  - `results/aufgabe03/<run_id>_run_local*_map.yaml`
  - `results/aufgabe03/<run_id>_run_local*_waypoints.csv`
  - `results/aufgabe03/<run_id>_run_local*_detected_obstacles.csv`
- Check waypoint/replan columns:
  `status`, `reached_count`, `min_scan_range_m`, `p05_scan_range_m`,
  `amcl_var_x`, `amcl_var_y`, `replan_count`, `run_local_replan_count`,
  `run_local_map_yaml`, `run_local_waypoints_csv`,
  `run_local_update_rejected_reason`, and `final_status_reason`.
- Evidence rule: completed waypoint rows without replan diagnostic columns
  support static waypoint navigation, not active obstacle map-update claims.

### Aufgabe 04 Single-Segment Real Parkour Run

- Use when collecting evidence for one robot driving one preplanned station
  route segment.
- Start from `docs/setups/aufgabe04_real_parkour_checklist.md`.
- The current runner is
  `scripts/aufgabe04/navigation/run_single_station_segment.py`.
- Required preflight evidence:
  route CSV/diagnostics validation, resolved namespace/topic/frame printout,
  fresh `/scan`, `/odom`, AMCL when used, fresh `map -> base_footprint` and
  `odom -> base_footprint`, `use_sim_time=false`, and fail-closed `/cmd_vel`
  ownership/Nav2 handoff checks.
- Dry run first:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --dry-run \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

- Real run must be wrapped. Keep bundle options matched to runner options:

```bash
scripts/common/run_with_bundle.sh run_001 -- \
  python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
    --leg-index 1 \
    --route-csv results/aufgabe04/routes/station_route.csv \
    --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json \
    --preflight-json results/real_runs/run_001/aufgabe04_preflight.json
```

- Expected report evidence remains under `results/aufgabe04/`; raw/debug
  evidence is under `results/real_runs/<run_id>/`.
- Do not treat zero-length legs as motion evidence. They require `--allow-noop`
  and should log `motion_published=false`.

## Post-Run Validation

Use relevant commands:

- safety stop first for real runs
- `ros2 bag info <bag_path>`
- `tail -n 2 results/aufgabe02/real_scripted_drive_runs.csv`
- `tail -n 2 results/aufgabe02/real_start_pose_checks.csv`
- `tail -n 2 results/aufgabe02/scripted_drive_runs.csv`
- `tail -n 2 results/aufgabe02/latest_tracker_pose.csv`
- `tail -n 3 results/aufgabe03/aufgabe03_waypoint_follow_runs.csv`
- `tail -n 3 results/aufgabe03/aufgabe03_arena_prior_two_stage_runs.csv`
- `python3 scripts/aufgabe02/analyze_probabilistic_endpoint_model.py`
- `python3 scripts/aufgabe02/build_motion_primitives_model.py`
- `python3 scripts/aufgabe03/analyze_waypoint_follow_runs.py`

Before relying on analysis scripts, inspect default input paths and run filters.

## Likely Failure Cases

Tailor to the selected protocol:

- wrong working directory
- missing ROS/TurtleBot environment setup
- `/cmd_vel` or `/odom` missing
- `/scan`, `/amcl_pose`, `/initialpose`, or `/navigate_to_pose` missing
- wrong `ROS_DOMAIN_ID` or network issue
- wrong `LDS_MODEL` for the connected LiDAR
- stale or missing `results/aufgabe02/latest_tracker_pose.csv`
- tracker running on a different host/path than the run script reads
- `valid_pose` false or not enough markers detected
- wrong camera stream, bad HSV thresholds, or bad calibration
- start-pose gate timeout
- existing bag output blocks rerun
- bag recording exits early
- `RUN_SPEED` or `RUN_DISTANCE` mistaken for actual motion control
- robot keeps moving after interruption
- analysis script excludes new runs due to hard-coded filters
- stale TF during waypoint following
- active obstacle replan enabled but missing map/path artifacts
- mirrored arena prior from mismatched forced heater/clean wall flags
- unexpected startup route change from initial run-local map replanning
- persistent scan blockage after the replan budget is exhausted
- RViz map/path debug publishers adding DDS/CPU timing load during real motion
