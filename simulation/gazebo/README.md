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
```

The spawn helper generates a simulation-only SDF under `/tmp` with a valid
80-degree pinhole horizontal field of view. It does not edit the installed
TurtleBot model or affect the real robot camera configuration. Gazebo publishes
an uncompressed image, so use the explicitly simulation-only raw-image option
below; real-robot commands continue to use `--compressed-image-topic`.

The two-stage contract is:

```text
LiDAR stand centre + current robot pose
  -> orientation-blind pre-approach
  -> camera QR/basic-side classification + stand-axis estimate
  -> final QR-facing pose
  -> final collision-aware route
```

At the stationary pre-approach pose, have the camera viewer continuously write
validated evidence (replace the four coordinates with the detected stand centre
and reached pre-approach pose):

```bash
python3 scripts/aufgabe04/perception/debug/stand_axis_viewer.py \
  --sim-raw-image-topic /camera/image_raw \
  --axis-source edges --stand-face-size-m 0.078 \
  --camera-fx-px <fx> --camera-fy-px <fy> \
  --robot-x <preapproach_x> --robot-y <preapproach_y> \
  --stand-x <detected_x> --stand-y <detected_y> \
  --observation-output-json results/aufgabe04/detected_stations/latest_camera_observation.json
```

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
