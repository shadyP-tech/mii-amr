# Aufgabe 04 Logistics Runbook

This runbook tracks the implementation order and evidence split for Aufgabe 04.
Keep logistics logic mostly pure. Add ROS motion only behind dry-run/preflight
gates, and wrap every physical run in a debug bundle.

Initial implementation order:

1. Pure QR parsing and station ordering.
2. Pure station map and route selection.
3. Dry-run mission state machine.
4. Navigation adapter around Aufgabe 03 planning.
5. Single-segment station-route execution behind strict preflight.
6. ROS camera integration, multi-segment missions, and two-robot operation only
   after single-robot dry-run and real-run evidence exists.

Dry-run artifact layout:

- Generated station layouts: `results/aufgabe04/layouts/`
- Dry-run station routes and diagnostics: `results/aufgabe04/routes/`
- Mission/evidence logs remain grouped by purpose under `results/aufgabe04/`
  or a more specific subfolder when a feature starts producing repeated files.
- Raw physical-run debug bundles go under `results/real_runs/<run_id>/`.

## Current Single-Segment Navigation Slice

Current route generation remains dry-run only:

```bash
python3 scripts/aufgabe04/navigation/run_station_route.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --stations A,B,C \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

Before physical motion, run the single-segment dry run:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --dry-run \
  --leg-index 1 \
  --route-csv results/aufgabe04/routes/station_route.csv \
  --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json
```

The dry run must pass route CSV validation, diagnostics cross-checking, speed
limits, resolved namespace/topic/frame checks, sensor freshness, TF/localization
freshness, `/cmd_vel` ownership, and Nav2 handoff checks.

For physical motion, wrap the command:

```bash
scripts/common/run_with_bundle.sh run_001 -- \
  python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
    --leg-index 1 \
    --route-csv results/aufgabe04/routes/station_route.csv \
    --diagnostics-json results/aufgabe04/routes/station_route_diagnostics.json \
    --preflight-json results/real_runs/run_001/aufgabe04_preflight.json
```

The bundle captures raw diagnostics only. It never publishes motion and does
not replace the runner's strict preflight or typed `RUN` confirmation.

If using namespaces, pass matching namespace/topic/frame options to
`run_with_bundle.sh` and `run_single_station_segment.py`; otherwise the bundle
may capture evidence for the wrong robot.

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
  --output-jsonl results/aufgabe04/detected_stations/stand_observations.jsonl
```

The observer uses `LaserScan.header.frame_id` as the source frame and requires a
fresh timestamped TF transform into `map`. It drops observations when frame or TF
provenance is missing or stale.

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
  --require-map-hash \
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
