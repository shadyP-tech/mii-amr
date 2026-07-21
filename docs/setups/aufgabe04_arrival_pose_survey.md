# Aufgabe 04 Simulation Arrival-Pose Survey

This simulation-only workflow separates stand inspection from the later
logistics route.

During the survey, the robot may drive only to camera/LiDAR inspection poses.
When the synchronized silhouette estimator commits an axis, the perpendicular
arrival pose on the robot-facing side is validated and written to a catalog.
That pose is **not** installed as the next live motion target. The current
survey leg ends with a zero-velocity command instead.

After every candidate is resolved, a separate planner freezes the catalog,
computes collision-checked directed A* costs between the stored arrival poses,
and uses exact Held-Karp optimization to create the full route. The existing
immediate-approach behavior remains the default outside this explicitly
selected simulation workflow.

## Candidate Input

Create one JSON file containing the stable LiDAR candidate IDs and centers.
For example, save this as
`results/aufgabe04/detected_stations/sim_candidates.json` and replace the
coordinates with the confirmed clusters from the current randomized world:

```json
{
  "candidates": [
    {
      "candidate_uid": "detected_stand_00",
      "stand_id": "detected_stand_00",
      "x_m": -0.395,
      "y_m": -0.415
    },
    {
      "candidate_uid": "detected_stand_01",
      "stand_id": "detected_stand_01",
      "x_m": 0.420,
      "y_m": 0.510
    }
  ]
}
```

Candidate IDs must be unique and stable for the entire survey. Use a new
catalog path and session ID after changing the Gazebo world, occupancy map, or
candidate set; provenance mismatches and conflicting retries fail closed.

## Phase 1: Survey Without Visiting the Computed Arrival Poses

With Gazebo and the camera-equipped Burger already running:

```bash
cd /workspace/mii-amr

source /opt/ros/humble/setup.bash
source /opt/tb3_src_ws/install/setup.bash
export ROS_DOMAIN_ID=31
export TURTLEBOT3_MODEL=burger

python3 scripts/aufgabe04/simulation/run_arrival_pose_survey.py \
  --candidates-json results/aufgabe04/detected_stations/sim_candidates.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --world simulation/gazebo/worlds/aufgabe04_stands.world \
  --output-dir results/aufgabe04/arrival_survey/session_001 \
  --catalog results/aufgabe04/detected_stations/arrival_pose_catalog_session_001.json \
  --catalog-id arrival_survey_session_001 \
  --session-id gazebo_session_001 \
  --map-frame odom \
  --initial-start-x 1.550 \
  --initial-start-y -0.600 \
  --initial-start-yaw 2.996
```

For each candidate the coordinator starts the synchronized camera/LiDAR
observer, publishes only acquisition/sampling routes, runs the dynamic follower,
waits for a committed silhouette axis, stores the exact future arrival pose,
and ends that survey leg. Existing catalog records are resumable and skipped
only when their map, world, and session provenance matches.

The authoritative result is the catalog JSON. Each record contains:

- the stable candidate and stand IDs;
- LiDAR stand center, radius, and uncertainty;
- canonical silhouette axis, confidence, sample count, and observation time;
- selected robot-facing normal and evidence provenance;
- exact perpendicular arrival and terminal-corridor entry poses;
- map/collision validation and source observation ancestry.

## Phase 2: Freeze and Optimize the Full Route

Choose the start pose for the later logistics route. If the robot will be reset
to the bottom-right corner before execution, use that reset pose here:

```bash
python3 scripts/aufgabe04/navigation/plan_arrival_catalog_route.py \
  --catalog results/aufgabe04/detected_stations/arrival_pose_catalog_session_001.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --world simulation/gazebo/worlds/aufgabe04_stands.world \
  --session-id gazebo_session_001 \
  --map-frame odom \
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

## Phase 3: Validate and Execute the Frozen Route

Dry-run a leg first while Gazebo topics are available:

```bash
python3 scripts/aufgabe04/navigation/run_single_station_segment.py \
  --route-csv results/aufgabe04/routes/optimized_arrival_route_session_001.csv \
  --diagnostics-json results/aufgabe04/routes/optimized_arrival_route_session_001_diagnostics.json \
  --leg-index 0 \
  --map-frame odom \
  --localization-source tf \
  --allow-sim-time \
  --dry-run \
  --allowed-cmd-vel-publisher /behavior_server \
  --allowed-cmd-vel-publisher /velocity_smoother
```

Then execute every optimized leg in sequence:

```bash
python3 scripts/aufgabe04/navigation/run_detected_stand_exploration_sim.py \
  --route-csv results/aufgabe04/routes/optimized_arrival_route_session_001.csv \
  --diagnostics-json results/aufgabe04/routes/optimized_arrival_route_session_001_diagnostics.json \
  --run-id-prefix arrival_route_session_001 \
  --map-frame odom
```

The final corridor waypoints are protected from CSV thinning. The static
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
