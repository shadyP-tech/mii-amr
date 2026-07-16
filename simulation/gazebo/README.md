# Aufgabe 04 Gazebo stands

This directory contains a portable Gazebo Classic world with static station
stands that match the physical green QR stands: four-foot green base, stem,
green head board, white QR panel, and black QR modules.

For Gazebo, the stand is uniformly scaled to the TurtleBot3 Burger envelope:
its top is 0.20 m above the floor. Every position and dimension, including the
four-foot base and QR panel, uses the same scale factor, so this robot-height
variant keeps the proportions of the physical reference stand.

The world is generated from a truth layout containing each hidden QR yaw. Do
not give that yaw to the orientation-blind pre-approach planner. It uses only a
detected stand centre; camera evidence resolves the QR side afterward.

Generate or refresh the world from the checked-in `A/B/C` layout:

```bash
cd /Users/stephpark/Documents/stephsWorld/mii-amr
python3 -m scripts.aufgabe04.simulation.generate_gazebo_world
```

Start the static world:

```bash
gazebo --verbose simulation/gazebo/worlds/aufgabe04_stands.world
```

The world intentionally does not spawn a TurtleBot. Start the TurtleBot3
Gazebo launch used by your ROS 2 Humble installation with this world as its
world argument, or spawn a robot separately. The stand models are static and
their base, stem, board, and arena walls have collision geometry; QR modules
are visual-only so they cannot create artificial collision hits.

Spawn the installed Burger camera model (Burger drivetrain plus a simulated
camera comparable to Waffle Pi):

```bash
source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash
scripts/aufgabe04/simulation/spawn_burger_camera.sh
ros2 topic list | grep -E 'image_raw|camera_info'
ros2 topic echo /gazebo_ground_truth --once
```

The spawn helper generates a simulation-only SDF under `/tmp` with a valid
80-degree pinhole horizontal field of view and a stamped, noise-free Gazebo
world pose on `/gazebo_ground_truth`. The standalone edge viewer combines that
pose with the image-nearest `/scan`: only map-wall rays confirmed by LiDAR are
removed, while closer foreground returns and a geometrically plausible raw
stand-head probe protect the selected stand corridor. Missing or stale evidence
disables suppression for that frame. This does not edit the installed
TurtleBot model or affect the real robot camera configuration. Gazebo publishes
an uncompressed image, so use the explicitly simulation-only raw-image option
below; real-robot commands continue to use `--compressed-image-topic`.

## GPT-oriented simulation debug bundles

Use the passive simulation wrapper when a navigation command should produce a
timestamp-aligned evidence bundle. Gazebo must already be running and `/clock`
must be visible. The wrapper itself never publishes motion; it only executes
the command supplied after `--`, captures evidence around it, and returns the
same exit status.

```bash
export ROS_DOMAIN_ID=31

RUN_ID=sim_debug_001
SEMANTIC_LOG=results/aufgabe04/run_events/${RUN_ID}.jsonl

scripts/aufgabe04/simulation/run_with_debug_bundle.sh "$RUN_ID" \
  --expected "follow leg 0, stop at the pre-approach pose, and face the stand" \
  --observed "" \
  --semantic-log "$SEMANTIC_LOG" \
  --perception-dir results/aufgabe04/debug/e2e_017_dynamic \
  -- python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
    --run-id "$RUN_ID" \
    --semantic-log "$SEMANTIC_LOG" \
    --allow-sim-time \
    --localization-source tf \
    --map-frame odom \
    --leg-index 0 \
    --route-csv results/aufgabe04/routes/detected_stand_exploration_route.csv \
    --diagnostics-json results/aufgabe04/routes/detected_stand_exploration_route_diagnostics.json
```

The default output is
`results/aufgabe04/simulation_debug_runs/<run_id>/` and contains:

- `manifest.json`: run intent, world, Git state, counts, paths, and warnings.
- `summary.md`: compact debugging question and evidence index for a GPT model.
- `timeline.jsonl`: merged telemetry, runner events, and conservative derived
  observations such as obstacle-threshold crossings and angular oscillation.
- `telemetry.jsonl`: 5 Hz odometry, commanded velocity, minimum LiDAR range,
  and Gazebo model ground truth when `/gazebo/model_states` is available.
- `frames/`: timestamped onboard frames and an automatically generated contact
  sheet. Pass `--overview-image-topic` to capture a second Gazebo camera.
- `perception/`: copied annotated frames, edges, face masks, rectangles, and
  ROI artifacts supplied with one or more `--perception-dir` options.
- `plots/trajectory.png` and `plots/velocity.png`.
- `rosbag/`, terminal output, capture logs, and before/after ROS graph
  snapshots for raw diagnosis.

Use `--frame-fps 0` or `--no-camera` when only telemetry is needed. Use
`--no-bag` for a lightweight exploratory run. Existing bundle directories are
never overwritten.

The two-stage contract is:

```text
LiDAR stand centre + current robot pose
  -> orientation-blind pre-approach
  -> camera QR/basic-side classification + stand-axis estimate
  -> final QR-facing pose
  -> final collision-aware route
```

During the simulated approach, have the camera viewer continuously project the
selected stand from synchronized odometry and write validated evidence. Replace
the two stand coordinates with the detected candidate; the robot pose comes
from `/odom` and must not be copied into static command-line arguments:

```bash
python3 scripts/aufgabe04/perception/debug/stand_axis_viewer.py \
  --sim-raw-image-topic /camera/image_raw \
  --axis-source edges --stand-face-size-m 0.06993 \
  --camera-fx-px 381.36246688 --camera-fy-px 381.36246688 \
  --camera-cx-px 320.5 --camera-cy-px 240.5 \
  --camera-forward-offset-m 0.076 \
  --odom-topic /odom --map-frame odom --base-frame base_footprint \
  --scan-topic /scan --use-lidar-distance \
  --lidar-bearing-source map-target \
  --stand-x <detected_x> --stand-y <detected_y> \
  --observation-output-json results/aufgabe04/detected_stations/latest_camera_observation.json
```

The cyan rectangle in the full camera view marks the exact projected ROI used
by the edge, face-mask, and rectangle diagnostics.

The measured Gazebo angle-estimation range is documented beside the standalone
viewer command in the
[perception debug runbook](../../scripts/aufgabe04/perception/debug/README.md#gazebo-exploratory-angle-estimation-range).
The exploratory sweep's low-error operating window is approximately
`0.224-0.374 m` camera optical depth with the stand face within
`+/-40 degrees`; this is a single-session simulated perception result, not a
navigation-safe standoff distance or real-camera validation.

Then consume that artifact to compute the final pose and yaw-aware route,
without reading the hidden layout yaw:

```bash
python3 scripts/aufgabe04/navigation/compute_qr_facing_pose.py \
  --observation-json results/aufgabe04/detected_stations/latest_camera_observation.json \
  --output results/aufgabe04/detected_stations/final_qr_pose.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --route-csv results/aufgabe04/routes/final_qr_route.csv \
  --diagnostics-json results/aufgabe04/routes/final_qr_route_diagnostics.json
```

The QR payloads are the station IDs (`A`, `B`, and `C`) encoded as Version 1-L
QR codes. The stand's local +x direction is its QR-facing direction, and the
layout yaw rotates that face toward the final approach side.

To use a new random layout, generate it first and then pass it to the world
generator:

```bash
python3 scripts/aufgabe04/navigation/generate_random_station_layout.py \
  --station-count 3 --seed 42 \
  --output results/aufgabe04/layouts/random_station_layout.json
python3 -m scripts.aufgabe04.simulation.generate_gazebo_world \
  --layout results/aufgabe04/layouts/random_station_layout.json
```
