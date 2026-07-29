# Aufgabe 04 LiDAR Observe-and-Plan Run

This run detects local stand candidates from `/scan`, waits for stationary
AMCL/TF readiness, selects one candidate, and writes a route. It does not
publish `/cmd_vel` and does not execute the route.

Keep the saved-map Nav2/AMCL launch active. Do not run Cartographer at the same
time because only one node may own `map -> odom`.

## Terminal A: robot bringup

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash 2>/dev/null || true

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02

ros2 launch turtlebot3_bringup robot.launch.py
```

## Terminal B: saved-map Nav2 and AMCL

```bash
cd /workspace/mii-amr
source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02

ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  use_sim_time:=False \
  map:=$PWD/maps/aufgabe03/arena_1p898x3p9_auto.yaml
```

Set the initial pose in RViz once. The automation command below creates both
its TF listener and LiDAR observer before requesting
`/request_nomotion_update`, so the stationary AMCL transform is not missed.

## Terminal C: automated observe and plan

```bash
cd /workspace/mii-amr
source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash

export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02

python3 scripts/aufgabe04/navigation/run_detected_stand_observe_plan.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --readiness-timeout-sec 30 \
  --observation-duration-sec 8 \
  --nomotion-refresh-sec 2 \
  --order confidence
```

The command creates a timestamped directory under `results/aufgabe04/`
containing:

- `stand_observations.jsonl`
- `candidate_snapshot.json`
- `layout.json` and `layout.csv`
- `route.csv` and `route_diagnostics.json`
- `exploration_state.json`
- `pipeline_summary.json`

`pipeline_summary.json` records the selected candidate, captured start pose,
route length, robot-facing terminal pose, physical-clearance calculation, and
`motion_published: false`. The default nominal Burger values are:

- robot radius: `0.105 m`
- certified tracking tube: `0.03 m`
- LiDAR stop distance: `0.20 m`
- static map inflation: `0.25 m`
- non-target candidate transit radius: `0.31 m`
- selected-stand approach offset: `0.32 m`

Replace the nominal mounting/clearance values with measured site values when
they differ.

## Terminal C: seal the selected pre-approach

Set `RUN_ROOT` to the exact timestamped directory printed by the preceding
command:

```bash
export RUN_ROOT=results/aufgabe04/real_explore_YYYYMMDD_HHMMSS

python3 scripts/aufgabe04/navigation/prepare_detected_stand_preapproach.py \
  --pipeline-root "$RUN_ROOT"
```

This creates a typed, non-simulation route plus its execution certificate
under `$RUN_ROOT/preapproach_execution/`. It binds the selected candidate
snapshot, map bundle, robot-facing terminal yaw, physical clearances, exact
route bytes, tracking tube, and command owner.

## Terminal C: live runner dry-run

This step performs artifact and live ROS preflight checks but never starts the
waypoint follower. The runner creates its AMCL subscriber and TF listener
first, requests `/request_nomotion_update`, waits for the transform chain to
become usable, and applies the `1.1 s` AMCL-only forward-TF allowance needed
by the standard `transform_tolerance: 1.0` configuration. Scan and odometry
timestamps retain the stricter `0.25 s` future limit.

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --route-csv "$RUN_ROOT/preapproach_execution/route.csv" \
  --diagnostics-json "$RUN_ROOT/preapproach_execution/route_diagnostics.json" \
  --route-certificate-json "$RUN_ROOT/preapproach_execution/route_certificate.json" \
  --candidate-snapshot "$RUN_ROOT/candidate_snapshot.json" \
  --leg-index 0 \
  --localization-source amcl \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_footprint \
  --scan-topic scan \
  --odom-topic odom \
  --amcl-topic amcl_pose \
  --cmd-vel-topic cmd_vel \
  --certified-route-tube-radius-m 0.03 \
  --min-obstacle-distance-m 0.20 \
  --preflight-json "$RUN_ROOT/preapproach_execution/preflight.json" \
  --semantic-log "$RUN_ROOT/preapproach_execution/dry_run_events.jsonl" \
  --results-csv "$RUN_ROOT/preapproach_execution/dry_run_results.csv" \
  --dry-run
```

Do not remove `--dry-run` until the physical parameters have been checked
against the robot, the dry-run passes, `/cmd_vel` ownership is unambiguous, the
arena is clear, and an operator is beside the robot with the physical stop and
a separate zero-Twist terminal ready. The later motion invocation still
requires an explicit typed `RUN`; the axis viewer remains diagnostic-only.

With the full TurtleBot3 Nav2 launch active, the dry-run may correctly reject
`/behavior_server` and `/velocity_smoother` as existing `/cmd_vel` publishers.
Do not bypass that result using `--allowed-cmd-vel-publisher`. Use a verified
command mux/guard or a launch profile in which the custom follower is the
single robot velocity owner while map server and AMCL remain active.
