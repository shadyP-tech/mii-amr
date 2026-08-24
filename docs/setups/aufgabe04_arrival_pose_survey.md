# Aufgabe 04 Simulation Arrival-Pose Survey

This simulation-only workflow separates stand inspection from the later
logistics route.

During the survey, the robot may drive only to camera/LiDAR inspection poses.
When the synchronized silhouette estimator commits an axis, the perpendicular
arrival pose on the robot-facing side is validated and written to a catalog.
That pose is **not** installed as the next live motion target. The current
survey leg ends with a zero-velocity command instead.

After every candidate is resolved, the coordinator freezes the catalog and
writes a survey manifest. A separate planner then computes collision-checked
directed A* transitions between the stored arrival poses. Survey routes may use
exact Held-Karp optimization; logistics routes must instead preserve the exact
server task order. The existing immediate-approach behavior remains the default
outside this explicitly selected simulation workflow.

The simulation-to-real status and blocking migration gate are documented in
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md).

## Candidate Input

Use the immutable candidate snapshot emitted by
`plan_detected_stand_exploration.py` from confirmed LiDAR observations. It
binds the complete candidate set, geometry/uncertainty, detector provenance,
and frozen map bundle. Pair it with a one-to-one station identity registry that
maps every candidate UID to one QR ID and one server station ID.

Candidate IDs must be unique and stable for the entire survey. Use
content-derived filenames or a new run directory after changing the Gazebo
world, occupancy map, detector result, identity mapping, or survey
configuration. Existing immutable paths accept byte-identical retries only.
The detector, observation producer, candidate snapshot, survey, and route
planner must all use the same planning frame. The examples below use `odom`,
so create the observations/snapshot with explicit `--map-frame odom` and
`--required-map-frame odom`; do not rely on the detector's `map` default.

`--candidates-json` remains available only with
`--allow-legacy-candidate-json`. That unsealed compatibility path does not emit
a survey manifest and is not acceptable migration evidence.

Create and review the identity registry before starting the survey. Repeat
`--mapping` exactly once for every candidate UID in the frozen snapshot:

```bash
python3 scripts/aufgabe04/stations/create_station_identity_registry.py \
  --candidate-snapshot results/aufgabe04/detected_stations/candidate_snapshot_HASH.json \
  --mapping detected_stand_00=A=station_A \
  --mapping detected_stand_01=B=station_B
```

The offline command rejects missing, unknown, or duplicate candidate, QR, and
server station IDs. By default it writes beside the snapshot as
`station_identity_registry_<full-content-hash>.json` and prints the exact path
and hashes. Supplying `--created-unix-sec` makes an intentional retry
byte-identical; an existing path is never replaced with different content.

## Phase 1: Survey Without Visiting the Computed Arrival Poses

With Gazebo and the camera-equipped Burger already running:

```bash
cd /workspace/mii-amr

source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash
export ROS_DOMAIN_ID=31
export TURTLEBOT3_MODEL=burger

python3 scripts/aufgabe04/simulation/run_arrival_pose_survey.py \
  --candidate-snapshot results/aufgabe04/detected_stations/candidate_snapshot_HASH.json \
  --station-identity-registry results/aufgabe04/detected_stations/station_identity_registry_HASH.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --world simulation/gazebo/worlds/aufgabe04_stands.world \
  --output-dir results/aufgabe04/arrival_survey/session_001 \
  --catalog results/aufgabe04/detected_stations/arrival_pose_catalog_session_001.json \
  --catalog-id arrival_survey_session_001 \
  --session-id gazebo_session_001 \
  --semantic-map-id arena_1p898x3p9_auto \
  --map-bundle-json results/aufgabe04/arrival_survey/session_001/map_bundle.json \
  --survey-manifest results/aufgabe04/arrival_survey/session_001/survey_manifest.json \
  --map-frame odom \
  --axis-sample-count 7 \
  --initial-start-x 1.550 \
  --initial-start-y -0.600 \
  --initial-start-yaw 2.996
```

For each candidate the coordinator starts the synchronized camera/LiDAR
observer, publishes only acquisition/sampling routes, runs the dynamic follower,
waits for a committed silhouette axis, stores the exact future arrival pose,
and ends that survey leg. Existing catalog records are resumable only when the
candidate identity, center, stand mapping, frozen map, world, and session still
match.

The authoritative result is the catalog JSON. Each record contains:

- the stable candidate and stand IDs;
- LiDAR stand center, radius, and uncertainty;
- canonical silhouette axis, confidence, sample count, and observation time;
- selected robot-facing normal and evidence provenance;
- exact perpendicular arrival and terminal-corridor entry poses;
- map/collision validation and source observation ancestry.

The sealed run also writes the frozen map descriptor and survey manifest. The
manifest links the exact map, candidate snapshot, world, survey configuration,
simulation calibration profile, and completed catalog. Hashes prove artifact
identity, not estimator accuracy.

## Phase 2A: Plan an Optimized Survey Route

Choose the start pose for the later logistics route. If the robot will be reset
to the bottom-right corner before execution, use that reset pose here:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/plan_arrival_catalog_route.py \
  --catalog results/aufgabe04/detected_stations/arrival_pose_catalog_session_001.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --world simulation/gazebo/worlds/aufgabe04_stands.world \
  --session-id gazebo_session_001 \
  --map-frame odom \
  --semantic-map-id arena_1p898x3p9_auto \
  --map-bundle-json results/aufgabe04/arrival_survey/session_001/map_bundle.json \
  --survey-manifest results/aufgabe04/arrival_survey/session_001/survey_manifest.json \
  --route-purpose survey \
  --start-x 1.550 \
  --start-y -0.600 \
  --start-yaw 2.996 \
  --route-csv results/aufgabe04/routes/optimized_arrival_route_session_001.csv \
  --diagnostics-json results/aufgabe04/routes/optimized_arrival_route_session_001_diagnostics.json \
  --visit-order-json results/aufgabe04/routes/optimized_arrival_route_session_001_visit_order.json \
  --pairwise-costs-json results/aufgabe04/routes/optimized_arrival_route_session_001_pairwise_costs.json \
  --catalog-snapshot-json results/aufgabe04/routes/optimized_arrival_route_session_001_catalog_snapshot.json
```

The command refuses incomplete catalogs, rejected candidates unless explicitly
allowed, map/frame mismatches, unreachable directed transitions, and candidate
counts above the exact optimization limit. It does not silently switch to the
opposite stand face, snap an arrival target, or substitute a heuristic route.

## Phase 2B: Plan the Exact Logistics Task Order

First run `run_logistics_mission.py --dry-run` to produce a freshly validated,
immutable task snapshot. Then plan with every sealed parent artifact. The task
snapshot is authoritative for station order; repeated `--fixed-station-order`
arguments are optional assertions and must match it exactly when supplied:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/plan_arrival_catalog_route.py \
  --catalog results/aufgabe04/detected_stations/arrival_pose_catalog_session_001.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --world simulation/gazebo/worlds/aufgabe04_stands.world \
  --session-id gazebo_session_001 \
  --map-frame odom \
  --semantic-map-id arena_1p898x3p9_auto \
  --map-bundle-json results/aufgabe04/arrival_survey/session_001/map_bundle.json \
  --candidate-snapshot results/aufgabe04/detected_stations/candidate_snapshot_HASH.json \
  --station-identity-registry results/aufgabe04/detected_stations/station_identity_registry_HASH.json \
  --survey-manifest results/aufgabe04/arrival_survey/session_001/survey_manifest.json \
  --task-snapshot results/aufgabe04/task_snapshots/task_MISSION_ID_HASH.json \
  --robot-id robot_1 \
  --route-purpose logistics \
  --start-x 1.550 \
  --start-y -0.600 \
  --start-yaw 2.996 \
  --route-csv results/aufgabe04/routes/task_route_session_001.csv \
  --diagnostics-json results/aufgabe04/routes/task_route_session_001_diagnostics.json \
  --route-certificate-json results/aufgabe04/routes/task_route_session_001_certificate.json \
  --planner-config-json results/aufgabe04/routes/task_route_session_001_planner_config.json \
  --route-bundle-json results/aufgabe04/routes/task_route_session_001_bundle.json \
  --mission-plan-manifest results/aufgabe04/routes/task_route_session_001_manifest.json
```

By default, planning rejects a snapshot more than 30 seconds old or timestamps
more than 2 seconds in the future. Refresh the dry-run snapshot instead of
relaxing these gates unless the clocks are known to require a bounded override.
The planner resolves semantic station IDs through the registry, plans only the
required transitions, refuses reordering, and writes an immutable mission-plan
manifest. It also persists hash-named planner-configuration and route-bundle
descriptors and prints their exact paths. Logistics planning requires the
surveyed catalog to already be frozen and never rewrites that source catalog.

## Phase 3: Admit One Task-Ordered Leg in Simulation

Dry-run a leg first while Gazebo topics are available:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/run_single_station_segment.py \
  --route-csv results/aufgabe04/routes/task_route_session_001.csv \
  --diagnostics-json results/aufgabe04/routes/task_route_session_001_diagnostics.json \
  --route-certificate-json results/aufgabe04/routes/task_route_session_001_certificate.json \
  --route-bundle-json results/aufgabe04/routes/task_route_session_001_bundle.json \
  --planner-config-json results/aufgabe04/routes/task_route_session_001_planner_config.json \
  --mission-plan-manifest results/aufgabe04/routes/task_route_session_001_manifest.json \
  --survey-manifest results/aufgabe04/arrival_survey/session_001/survey_manifest.json \
  --runtime-map-bundle-json results/aufgabe04/arrival_survey/session_001/map_bundle.json \
  --runtime-environment simulation/gazebo/worlds/aufgabe04_stands.world \
  --candidate-snapshot results/aufgabe04/detected_stations/candidate_snapshot_HASH.json \
  --station-identity-registry results/aufgabe04/detected_stations/station_identity_registry_HASH.json \
  --arrival-pose-catalog results/aufgabe04/detected_stations/arrival_pose_catalog_session_001.json \
  --task-snapshot results/aufgabe04/task_snapshots/task_MISSION_ID_HASH.json \
  --robot-id robot_1 \
  --leg-index 0 \
  --map-frame odom \
  --localization-source tf \
  --allow-sim-time \
  --dry-run \
  --allowed-cmd-vel-publisher /behavior_server \
  --allowed-cmd-vel-publisher /velocity_smoother
```

This validates the exact task, robot, station/candidate order, route bundle,
certificate, planning frame, frozen map descriptor, and task-plan freshness.
The same validation is repeated immediately before motion.

The optimized survey route is only a geometry demonstration; it is not a
logistics mission. To dry-run one such survey leg, explicitly add
`--allow-unbound-survey-simulation-route` with `--allow-sim-time`. Never use
that escape hatch for task execution.

Do not use the multi-leg wrapper for a logistics mission yet. It does not
produce post-arrival QR confirmations or persist `MissionController` dispatch
state, so it cannot prove ordered task completion. Once that adapter exists,
it must pass the same mission, route-bundle, certificate, and runtime-map inputs
for every sequential leg. The old survey-only demonstration command is:

```bash
python3 scripts/aufgabe04/navigation/entrypoints/run_detected_stand_exploration_sim.py \
  --route-csv results/aufgabe04/routes/optimized_arrival_route_session_001.csv \
  --diagnostics-json results/aufgabe04/routes/optimized_arrival_route_session_001_diagnostics.json \
  --run-id-prefix arrival_route_session_001 \
  --map-frame odom
```

The route planner also writes a content-hashed execution certificate and binds
its path/hash into diagnostics. The single-segment runner validates that
certificate before a static physical route and the follower checks the live
pose/pursuit chord against the certified route tube. The final corridor
waypoints are protected from CSV thinning. The static
`catalog_face_approach` route kind enables the existing follower's terminal
heading corridor so each selected stored pose is approached along its face
normal and ends facing the stand.

## Offline Verification

```bash
python3 -m unittest discover -s tests/aufgabe04
```

These tests are ROS-free. Passing them verifies data contracts, geometry,
catalog integrity, fixed-face path planning, route optimization, artifact
compatibility, and the survey-completion handoff; it is not evidence of a live
Gazebo drive.
