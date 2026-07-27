# Aufgabe 04 Logistics Runbook

This runbook tracks the implementation order and evidence split for Aufgabe 04.
Keep logistics logic mostly pure. Add ROS motion only behind dry-run/preflight
gates, and wrap every physical run in a debug bundle.

The authoritative implementation status, sealed artifact chain, and explicit
hardware migration blockers are in
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md). The repository
is **not currently cleared for real end-to-end logistics or two-robot motion**.

Initial implementation order:

1. Pure QR parsing and station ordering.
2. Pure station map and route selection.
3. Dry-run mission state machine.
4. Navigation adapter around Aufgabe 03 planning.
5. Single-segment station-route execution behind strict preflight.
6. ROS camera integration, multi-segment missions, and two-robot operation only
after single-robot dry-run and real-run evidence exists.

The simulation workflow that surveys future perpendicular stand-arrival poses
is documented in
[`aufgabe04_arrival_pose_survey.md`](aufgabe04_arrival_pose_survey.md).
The dedicated hardware-profile, passive real survey, and unloaded single-leg
adapters are documented in
[`aufgabe04_real_pipeline.md`](aufgabe04_real_pipeline.md).

Current pure foundations include immutable map/candidate/survey/task/mission
artifacts, exact server-order planning, strict QR evidence and mission-state
contracts, route-tube certificates, carrier/custody models, and fenced
station/conflict-zone permits. Do not interpret “implemented” here as “wired to
ROS”: strict QR event production, mission-to-follower dispatch, the independent
command guard node/mux, carrier sensing, execution-manifest generation, and the
fleet coordinator transport remain incomplete.

Dry-run artifact layout:

- Generated station layouts: `results/aufgabe04/layouts/`
- Dry-run station routes and diagnostics: `results/aufgabe04/routes/`
- Mission/evidence logs remain grouped by purpose under `results/aufgabe04/`
  or a more specific subfolder when a feature starts producing repeated files.
- Raw physical-run debug bundles go under `results/real_runs/<run_id>/`.

## Legacy Offline Station-Route Slice

The older station-map generator remains useful for offline visualization only:

```bash
python3 scripts/aufgabe04/navigation/run_station_route.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --stations A,B,C \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

These legacy CSVs have no sealed route kind, certificate, or mission root and
are intentionally rejected by `run_single_station_segment.py`. Do not use them
for motion. For simulation admission, follow the complete candidate → survey →
task → mission workflow in `aufgabe04_arrival_pose_survey.md`. Physical mission
motion remains blocked by `aufgabe04_sim_to_real_gate.md`.

Debug bundles capture evidence only and never make a legacy route executable.

## Tests

```bash
python3 -m unittest discover tests/aufgabe04
python3 -m unittest tests.test_run_with_bundle
```

Default tests must remain ROS-free.

## Observe-Only LiDAR Stand Discovery

The first detected-station slice is split into observe-only perception, ROS-free
artifact planning, and the existing gated segment runner. The observer never
publishes `cmd_vel`.

Observe live LiDAR stand candidates and write provenance-rich evidence:

```bash
python3 scripts/aufgabe04/perception/stand_explorer_node.py \
  --scan-topic scan \
  --map-frame map \
  --base-frame base_footprint \
  --localization-source amcl \
  --map-yaml maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --output-jsonl results/aufgabe04/detected_stations/stand_observations.jsonl
```

The observer uses `LaserScan.header.frame_id` as the source frame and requests
the transform into `map` at the exact scan stamp. A bounded nonblocking queue
lets the TF listener catch up without falling back to the latest transform. It
drops observations when the scan/TF stamps are zero, stale, future-dated, too
far apart, or inconsistent with the recorded ROS clock mode.

Create an explicit confirmation receipt before route planning. This is the
manual/QR gate that turns one unique confirmed LiDAR stand into a station
identity; it does not move the robot:

```bash
python3 scripts/aufgabe04/navigation/create_detected_station_confirmation.py \
  --observations-jsonl results/aufgabe04/detected_stations/stand_observations.jsonl \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --station-id A \
  --confirmation-source operator \
  --operator-confirmed \
  --output-json results/aufgabe04/detected_stations/first_detected_station_confirmation.json
```

Create first-station layout and route artifacts from confirmed observations:

```bash
python3 scripts/aufgabe04/navigation/read_current_amcl_pose.py \
  --amcl-topic amcl_pose \
  --map-frame map \
  --max-age-sec 2.0
```

Use the printed `--start-x`, `--start-y`, and `--start-yaw` arguments in the
planner command:

```bash
python3 scripts/aufgabe04/navigation/plan_first_detected_station.py \
  --observations-jsonl results/aufgabe04/detected_stations/stand_observations.jsonl \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --start-x <amcl_x> \
  --start-y <amcl_y> \
  --start-yaw <amcl_yaw> \
  --confirmation-json results/aufgabe04/detected_stations/first_detected_station_confirmation.json \
  --layout-json results/aufgabe04/detected_stations/first_detected_station_layout.json \
  --layout-csv results/aufgabe04/detected_stations/first_detected_station_layout.csv \
  --route-csv results/aufgabe04/routes/first_detected_station_route.csv \
  --diagnostics-json results/aufgabe04/routes/first_detected_station_route_diagnostics.json
```

Physical motion remains a separate step through
`scripts/aufgabe04/navigation/run_single_station_segment.py --dry-run` and then
the bundle-wrapped real runner with typed `RUN`.

## Offline Stand-Axis Analysis

Stand-axis estimation is an analysis-only evidence path. It must not feed route
generation or physical motion until exported-scan evidence shows that the axis is
reliable.

Use `scripts/aufgabe04/perception/stand_axis_analysis.py` from Python with plain
JSON/CSV exports. The analyzer accepts point samples or range arrays, clusters
them with the existing LiDAR stand detector, estimates an optional cluster axis,
compares it with manually entered `truth_axis_rad`, and writes CSV metrics such
as point count, width, estimated axis, angular error, confidence, and
usable/not-usable reason.

Keep this path offline:

- no ROS bag parsing inside the analyzer
- no `rclpy` or `sensor_msgs`
- no `/cmd_vel`
- no station planner or route-generation dependency

If a ROS bag or `sensor_msgs/LaserScan` exporter is needed later, keep it as a
separate adapter that writes plain JSON/CSV for the analyzer.

## Debug-Only Stand-Axis LiDAR ROI

The live stand-axis viewer may optionally write observe-only LiDAR ROI debug
rows to:

```text
results/aufgabe04/stand_axis_lidar_roi/stand_axis_lidar_roi_observations.jsonl
```

This artifact is for diagnosing camera-rectangle-to-LiDAR distance selection in
`scripts/aufgabe04/perception/debug/stand_axis_viewer.py`. It is not real-parkour
evidence and must not feed `plan_first_detected_station.py`,
`run_single_station_segment.py`, station routing, collision stops, mission
state, or physical motion.

The default LiDAR mode remains the fixed scan-frame cone. The optional
`--lidar-bearing-source image-center` mode maps the detected rectangle center
through processed-image intrinsics and an optional measured camera-to-LiDAR yaw
offset. Treat that mapping as a calibration assumption until it has been
validated on the actual TurtleBot.

Gazebo additionally supports `--lidar-bearing-source map-target`. That mode is
simulation-only: it synchronizes `/camera/image_raw` with `/odom`, derives the
LaserScan bearing from the selected stand's map coordinates, and projects the
camera ROI with the simulated camera extrinsics. It fails closed on stale or
mismatched odometry and never changes the real-camera fixed/image-center path.
