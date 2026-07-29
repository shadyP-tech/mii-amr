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
route length, and `motion_published: false`.

The resulting exploration CSV is not a sealed real-motion route. Do not pass
it to `run_single_station_segment.py` by weakening route-kind or simulation
gates. A physical approach requires the real hardware/site profile, passive
survey/catalog, certified route artifacts, dry-run preflight, and explicit
operator motion confirmation.
