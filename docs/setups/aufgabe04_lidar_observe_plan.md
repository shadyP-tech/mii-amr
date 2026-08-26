# Aufgabe 04 LiDAR Observe-and-Plan Run

This run detects local stand candidates from `/scan`, waits for stationary
AMCL/TF readiness, selects one candidate, and writes a route. It does not
publish `/cmd_vel` and does not execute the route.

## Multi-viewpoint coverage survey

The one-shot command later in this document is retained for focused
one-candidate checks. It is not evidence that the whole arena was surveyed.
For arena-wide discovery, use the map-derived coverage survey:

```text
plan two staggered rails
  -> reach one survey viewpoint
  -> stop and collect an exact-time-TF LiDAR epoch
  -> persist negative or positive scan evidence
  -> fuse stable candidate IDs and provisional keepouts
  -> replan the next leg
  -> repeat until the coverage gate passes
  -> inspect pending candidates with the stopped camera
```

The LDS is a 360-degree scanner. Translating between the staggered rails is
what creates new lines of sight; rotating repeatedly at the initial pose does
not resolve geometric occlusion.

### 1. Create the coverage plan

Use the current localized base pose for `--start-x`, `--start-y`, and
`--start-yaw`. This command is ROS-free and motion-free:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/plan_stand_coverage_survey.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --planning-frame map \
  --start-x CURRENT_MAP_X \
  --start-y CURRENT_MAP_Y \
  --start-yaw CURRENT_MAP_YAW \
  --survey-id stand_coverage_001 \
  --output-dir results/aufgabe04/stand_coverage_001
```

The default planner uses two staggered rails, `0.90 m` stop spacing, a
conservative `1.35 m` visibility model, `0.25 m` static inflation, and a
`95%` coverage gate. It writes:

- `coverage_plan.json`: immutable planned viewpoints and per-viewpoint
  line-of-sight cells;
- `coverage_progress.json`: visited viewpoints;
- `stand_registry.json`: stable candidates and their statuses;
- `legs/leg_000_route.csv` plus diagnostics: motion-free A* planning evidence;
- `survey_summary.json`: coverage and unresolved-candidate gates.

For the measured small arena, an explicit minimal two-stop plan can be
requested without weakening the visibility model:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/plan_stand_coverage_survey.py \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --planning-frame map \
  --start-x CURRENT_MAP_X \
  --start-y CURRENT_MAP_Y \
  --start-yaw CURRENT_MAP_YAW \
  --survey-id stand_coverage_two_stop_001 \
  --output-dir results/aufgabe04/stand_coverage_two_stop_001 \
  --lane-count 1 \
  --stop-spacing-m 0.70 \
  --exact-inspection-point-count 2 \
  --expected-stand-count 5
```

The exact-two selector chooses two distinct centerline cells from the normal
dense plan by maximum union coverage and shared visibility. It still rejects
the plan unless the unchanged `95%` map-coverage gate passes. In
`execute-coverage-checkpoint` mode, the two-stop survey may finish as a
LiDAR-only checkpoint when exactly the expected number of non-rejected,
static-map-admitted candidates have valid confidence, hit, and at least one
known planned-viewpoint observation. The terminal evidence keeps provisional,
multi-view, camera-queue, confirmed, and rejected states separate. It sets
`camera_approach_authorized=false`, creates no candidate snapshot, and cannot
be resumed as a motion checkpoint.

Use the separate `execute-exact-two-camera` mode when the intended workflow is
two LiDAR stops followed by camera validation of all expected candidates. It
requires exactly two inspection points and exactly the physical-site stand
count, plus a content-hashed stand model with `environment=physical` and
`measurement_status=measured`. Missing or provisional geometry fails before
coverage motion. It writes a content-hashed handoff that binds the terminal coverage
checkpoint, LiDAR admission, live registry, and frozen candidate snapshot.
Multi-view `pending_camera` candidates and eligible single-view `provisional`
candidates remain distinct in that evidence; a provisional candidate can be
resolved only by the bound camera phase. The generic `execute-full` admission
is unchanged and still accepts only its multi-view `pending_camera` queue.

The generated leg is deliberately marked `motion_authorized: false`. Do not
feed it directly to the real-robot segment runner. Real survey execution still
needs a separately reviewed route certificate/admission wrapper, dry-run,
fresh sensor/localization checks, exclusive velocity ownership, and the
physical stop precautions in the real parkour checklist.

### 2. At each stopped viewpoint, capture one observation epoch

After a separately authorized mechanism has placed and stopped the robot at
the `next_viewpoint_id` from `survey_summary.json`, capture a fresh epoch using
unique paths:

```bash
export SURVEY_ROOT=results/aufgabe04/stand_coverage_001
export VIEWPOINT_ID=survey_vp_001
mkdir -p "$SURVEY_ROOT/raw_epochs/$VIEWPOINT_ID"

python3 scripts/aufgabe04/perception/stand_explorer_node.py \
  --scan-topic scan \
  --amcl-topic amcl_pose \
  --map-frame map \
  --base-frame base_footprint \
  --localization-source amcl \
  --map-yaml maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --duration-sec 8 \
  --observation-id-scope "$VIEWPOINT_ID" \
  --output-jsonl "$SURVEY_ROOT/raw_epochs/$VIEWPOINT_ID/observations.jsonl" \
  --summary-json "$SURVEY_ROOT/raw_epochs/$VIEWPOINT_ID/observer_summary.json"
```

`observer_summary.json` is written even when no candidate was found. A
viewpoint counts as covered only when that receipt reports at least one
processed scan and its exact map bundle, planning frame, and observed scan
pose match the plan. The explicit viewpoint scope makes observation IDs unique
across the separate observer processes; never omit it when epochs will be
fused into one registry.

### 3. Fuse candidates and replan

```bash
python3 scripts/aufgabe04/navigation/entrypoints/record_stand_coverage_stop.py \
  --survey-root "$SURVEY_ROOT" \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --viewpoint-id "$VIEWPOINT_ID" \
  --observer-summary-json \
    "$SURVEY_ROOT/raw_epochs/$VIEWPOINT_ID/observer_summary.json"
```

This command:

- validates the stopped scan receipt and observation provenance;
- marks the viewpoint visited, including valid zero-candidate epochs;
- keeps a candidate provisional after only one viewpoint;
- promotes it to `pending_camera` after sufficient hits from two distinct
  viewpoints;
- rejects replayed observation IDs as fake viewpoint diversity;
- adds every non-rejected candidate as a keepout;
- writes a new A* leg to the next reachable unvisited viewpoint.

The survey is not complete merely because no candidate is pending. Summary
fields distinguish `lidar_coverage_complete` from
`camera_exploration_complete`; the legacy `exploration_complete` is an alias
for camera completion and therefore remains false at a successful LiDAR-only
checkpoint.

### 4. Record a stopped camera/operator decision

After inspecting a `pending_camera` candidate, create a small receipt:

```json
{
  "schema_version": 1,
  "survey_id": "stand_coverage_001",
  "candidate_uid": "survey_candidate_0001",
  "decision": "confirmed",
  "decision_source": "camera_evidence",
  "camera_evidence_path": "results/real_runs/EXAMPLE/camera_capture"
}
```

Then record it without invoking motion:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/record_stand_candidate_decision.py \
  --survey-root "$SURVEY_ROOT" \
  --decision-receipt-json candidate_decision.json
```

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

python3 scripts/aufgabe04/navigation/entrypoints/run_detected_stand_observe_plan.py \
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

python3 scripts/aufgabe04/navigation/entrypoints/prepare_detected_stand_preapproach.py \
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
python3 scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py \
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
