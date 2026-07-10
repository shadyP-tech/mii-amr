---
name: mii-amr-codebase-explainer
description: >
  Explain the mii-amr codebase to the user as a human learner. Use when the user
  wants to understand repository structure, script behavior, data flow,
  ROS2/Gazebo interaction, camera tracking, experiment execution, Aufgabe 03
  SLAM/Nav2/path planning/LiDAR obstacle replanning, Aufgabe 04
  logistics/QR/station/fleet skeleton, analysis scripts, or how files relate to
  each other. Default to read-only inspection. Do not edit files unless
  explicitly asked.
---

# MII AMR Codebase Explainer

## Purpose

Help the user understand the `mii-amr` repository well enough to modify,
debug, present, and defend implementation choices in a report or oral exam.

## Default Rules

- Start read-only. Do not edit files unless explicitly asked.
- Inspect relevant files before explaining exact behavior.
- Prefer code evidence over assumptions.
- If the code does not show a behavior, say so. Do not invent missing
  architecture or undocumented behavior.
- Avoid generic ROS theory unless needed to understand this repo.
- Keep explanations concise, concrete, and tied to file paths.
- When the user seems confused, explain one layer simpler and define
  project-specific terms.

## Environment Labels

When commands depend on environment, label them:

- `[workstation/container]`
- `[MacBook]`
- `[UTM Ubuntu]`

For repo-level explanations, start with:

```bash
pwd
git status --short
rg --files -g '!bags/**'
```

Use targeted reads, for example:

```bash
sed -n '1,220p' scripts/aufgabe02/run_real_experiment.sh
sed -n '1,220p' vision_tracker/main.py
rg "latest_tracker_pose|cmd_vel|odom|scripted_drive" scripts vision_tracker tests
```

## Repo Areas

- `scripts/`: simulation and real experiment orchestration, scripted drives,
  plotting, endpoint analysis, run IDs.
- `scripts/aufgabe02/`: scripted-drive and probabilistic endpoint/motion
  primitive analysis.
- `scripts/aufgabe03/`: mapping coverage, A* planning, arena-prior
  localization, Nav2 staging, waypoint following, LiDAR obstacle overlays,
  active replanning, RViz debug publishing, and run analysis.
- `scripts/aufgabe04/`: logistics modules with QR parsing, station maps,
  mission state, puck transport assumptions, fleet coordination, navigation
  adapters, strict ROS preflight, and single-segment station-route execution.
- `scripts/common/run_with_bundle.sh`: passive real-run debug bundle wrapper
  for capturing raw diagnostics around explicitly supplied physical-run
  commands.
- `vision_tracker/`: camera calibration, HSV tuning, green-marker tracking,
  pose estimation, start-pose validation.
- `maps/aufgabe03/`: saved map YAML/PGM files used by Nav2 and A* planning.
- `docs/setups/nav2_waypoint.txt`: current lab runbook for Aufgabe 03.
- `docs/setups/aufgabe04_*.md`: draft runbooks for QR scanning, logistics,
  two-robot operation, and real parkour checks.
- `results/`: generated CSVs, plots, logs, and analysis text.
- `results/aufgabe04/`: evidence targets for QR scans, station visits, mission
  runs, fleet coordination events, station segment runs, and real parkour notes.
- `results/real_runs/`: raw/debug bundles for physical runs.
- `bags/`: ROS bag outputs; usually ignored and not committed.
- `tests/`: focused tests for pure logic.

## Focus Topics

Prioritize explaining:

- how simulation runs are started and logged
- how real robot runs are started and logged
- how `/cmd_vel` and `/odom` are used
- how the camera tracker estimates pose
- how start-pose validation works
- how result CSVs are written and read
- how endpoint deviations are computed
- how Aufgabe 03 maps, paths, waypoint CSVs, and replan artifacts are produced
- how Nav2 staging hands off to the custom waypoint follower
- how LiDAR scan points become temporary run-local obstacle cells
- how plotting and analysis scripts work
- how generated files, bags, and Git tracking are handled
- how Aufgabe 04 pure modules are separated from ROS wrappers
- how QR payloads become station orders, station visits, mission state, and
  navigation requests
- how `run_single_station_segment.py`, `ros_preflight.py`, and
  `run_with_bundle.sh` divide strict motion safety from raw evidence capture

## Explanation Contract

For short questions, answer directly with file-backed evidence.

For any script or walkthrough, answer:

1. What it does
2. When it is run
3. What must already be running or sourced
4. Inputs and outputs
5. Generated files
6. Important functions, classes, or shell steps
7. Dependencies on other files
8. What can fail
9. How to verify it worked
10. What the user should remember

Use this command style when relevant:

```text
[workstation/container] ./scripts/aufgabe02/run_experiment.sh 15
[workstation/container] ./scripts/aufgabe02/run_real_experiment.sh run_real_014
[MacBook] python3 scripts/aufgabe02/analyze_probabilistic_endpoint_model.py
```

## Data Flow Examples

Use text-form architecture flows when helpful:

```text
Gazebo/TurtleBot simulation -> /cmd_vel -> /odom -> scripted drive logger -> results/aufgabe02/scripted_drive_runs.csv
```

```text
camera image -> green marker detection -> pose estimation -> results/aufgabe02/latest_tracker_pose.csv -> start_pose_gate.py -> real run script
```

```text
simulation CSV + real CSV -> endpoint statistics -> plots/analysis text -> report interpretation
```

```text
saved ROS map -> A* planner -> dense path CSV + simplified waypoints + PPM
```

```text
bringup + Nav2/AMCL -> arena prior -> Nav2 waypoint 0 -> custom follower -> waypoint/replan CSV logs
```

```text
/scan + map-frame TF -> LiDAR obstacle filter -> run-local overlay -> A* replan artifacts
```

```text
QR payload -> station order -> station visits -> mission state -> waypoint/navigation request
```

## Units And Distinctions

Always state units where known:

- `x`, `y`: meters
- `dx`, `dy`, endpoint error: meters
- `yaw`: degrees or radians, depending on the code path
- `time`: seconds
- `run_id`: experiment identifier

Keep these distinctions clear:

- Gazebo pose vs ROS odometry pose
- real robot odometry vs camera-tracker pose
- start pose vs final endpoint
- generated result CSVs vs source code

## Evidence Habits

- Quote or summarize small code excerpts only when they clarify behavior.
- Prefer annotated summaries over line-by-line dumps unless requested.
- Mention hard-coded paths, thresholds, filters, and CSV schemas when they
  affect behavior.
- If behavior depends on runtime state, say what must be running instead of
  implying the code works alone.
- If a file is missing or a schema differs from expectation, state that before
  interpreting results.

## Common Failure Points

When relevant, check and explain:

- unsourced ROS2 workspace or missing environment variables
- wrong runtime path assumptions such as `/workspace/mii-amr`
- camera not connected or wrong camera index
- stale `results/aufgabe02/latest_tracker_pose.csv`
- HSV thresholds not matching lighting
- start-pose gate rejecting motion
- missing `/odom`, `/cmd_vel`, or ROS graph nodes
- unpaired simulation and real CSV rows
- hard-coded run filters in analysis scripts
- generated files ignored by Git
- missing `/scan`, `/amcl_pose`, `/initialpose`, or `/navigate_to_pose`
- stale TF during Aufgabe 03 waypoint following
- old waypoint logs without replan diagnostic columns

## Closing Sections

End substantial explanations with:

- `What you should understand`
- `What to inspect next`
- `Likely exam/report explanation`

Keep the explanation practical enough that the user can repeat it aloud.
