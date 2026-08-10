# Aufgabe 04 Real-Robot Experiment Pipeline

The real experiment path reuses the simulation pipeline's immutable station,
catalog, route, and mission contracts, but replaces simulated sensing and
runtime assumptions with dedicated hardware adapters. It is not approval for a
loaded logistics mission or a two-robot run; see
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md).

## Module Boundary

| Module | Responsibility | Motion capability |
| --- | --- | --- |
| `real_robot/capture_camera_calibration.py` | Capture live `CameraInfo` and the measured `base <- camera_optical` TF | None |
| `real_robot/create_hardware_profile.py` | Seal topics, namespace, frames, site, calibration, footprint, and speed limits | None |
| `perception/stand_axis_image.py` | Stable compatibility façade and estimator orchestration | None |
| `perception/stand_axis/models.py` | Immutable estimator, point, support, and debug-artifact contracts | None |
| `perception/stand_axis/preprocessing.py` | Color-agnostic raw Canny extraction and topology-only gap recovery | None |
| `perception/stand_axis/stem_candidates.py` | Stem-anchored head localization and candidate construction | None |
| `perception/stand_axis/raw_support.py` | Four-side raw-edge support, common-side direction, and trapezoid refit | None |
| `perception/stand_axis/geometry.py` | Quadrilateral geometry, square-head pose estimation, and debug rendering | None |
| `perception/stand_axis/real_camera_profile.py` | Validate and resolve the offline-candidate real-camera edge recipe | None |
| `navigation/runtime_localization_reseal.py` | Classify the exact global-consistency zero/reseal contract and enforce its retry budget | None |
| `navigation/runtime_motion_authorization.py` | Bind the mission-level `RUN` to one exact, same-target runtime-localization recovery child and its fresh artifacts | None |
| `navigation/runtime_motion_consumption.py` | Atomically consume that exact child permit once and reject replay before follower motion | None |
| `navigation/mission_leg_motion_permit.py` | Bind one mission-level `RUN` to separately sealed routine coverage, candidate, and opposite-face child legs | None |
| `navigation/mission_leg_motion_consumption.py` | Atomically consume each exact routine-leg permit once immediately before motion | None |
| `navigation/coverage_candidate_admission.py` | Fail closed between LiDAR coverage and candidate approaches unless coverage and multi-view candidate evidence are complete | None |
| `real_robot/passive_viewpoint_node.py` | Synchronize image, scan, and exact-time TF; rectify the image; validate LiDAR/QR/silhouette evidence | None |
| `real_robot/run_autonomous_stand_exploration.py` | Plan and execute the unloaded center-corridor discovery, candidate inspection, and QR-facing pose catalog | Dry-run by default; explicit physical gate |
| `real_robot/prepare_passive_survey.py` | Produce immutable per-candidate observer and catalog-validation commands | None |
| `real_robot/finalize_passive_survey.py` | Freeze a complete real arrival catalog and write a real `SurveyManifest` | None |
| `real_robot/run_unloaded_segment.py` | Bind one certified route leg to the sealed hardware/site profile and existing preflight runner | Dry-run by default; explicit physical gate |

The shared `plan_synchronized_viewpoint.py` accepts
`--environment real --workflow-mode survey-only`. In real mode it cannot watch
or generate a dynamic motion route. It only validates an already committed
passive recommendation and updates the shared arrival catalog.

## 1. Inspect the Live ROS Interface

The SSH alias `turtle` must resolve from the workstation before remote
inspection. Use read-only commands first:

```bash
ssh turtle 'hostname; printenv ROS_DOMAIN_ID; ros2 topic list; ros2 node list'
ssh turtle 'ros2 topic info /scan -v; ros2 topic info /camera/camera_info -v'
ssh turtle 'ros2 run tf2_ros tf2_echo base_footprint camera_optical_frame'
```

Substitute the robot's actual namespace, camera topics, and optical frame in
all later commands. Do not infer them from Gazebo names.

## 2. Seal Calibration and Hardware Inputs

Create a physical-site descriptor whose filename stem is its stable site ID.
It should identify the measured parkour revision, station placement procedure,
map acquisition, and date. The pipeline binds its exact bytes:

```json
{
  "schema_version": 1,
  "physical_site_id": "aufgabe04_lab_v1",
  "description": "Measured Aufgabe 04 parkour, revision 1",
  "map_measurement": "replace with the measurement record",
  "station_setup": "replace with the station placement record"
}
```

Capture the calibration that is currently published by the real camera. This
is a capture step, not proof that the calibration itself is accurate:

```bash
python3 scripts/aufgabe04/real_robot/capture_camera_calibration.py \
  --namespace robot1 \
  --camera-info-topic camera/camera_info \
  --base-frame base_footprint \
  --camera-optical-frame camera_optical_frame \
  --calibration-id robot1_camera_20260727 \
  --source "measured calibration report <identifier>" \
  --output results/aufgabe04/real/profiles/robot1_camera_20260727.json
```

Measure the physical robot radius, LiDAR-to-base radial offset, and conservative
unloaded speed limits. Then seal them with the resolved runtime interface:

```bash
python3 scripts/aufgabe04/real_robot/create_hardware_profile.py \
  --profile-id robot1_unloaded_20260727 \
  --robot-id robot1 \
  --namespace robot1 \
  --scan-topic scan \
  --odom-topic odom \
  --cmd-vel-topic cmd_vel \
  --amcl-topic amcl_pose \
  --compressed-image-topic camera/image_raw/compressed \
  --camera-info-topic camera/camera_info \
  --map-frame map \
  --odom-frame odom \
  --base-frame base_footprint \
  --scan-frame base_scan \
  --localization-source amcl \
  --physical-site docs/setups/aufgabe04_lab_v1.json \
  --physical-site-id aufgabe04_lab_v1 \
  --camera-calibration results/aufgabe04/real/profiles/robot1_camera_20260727.json \
  --robot-radius-m <measured_radius> \
  --scan-origin-to-base-offset-m <measured_offset> \
  --max-linear-speed-mps <validated_unloaded_limit> \
  --max-angular-speed-radps <validated_unloaded_limit> \
  --output results/aufgabe04/real/profiles/robot1_unloaded_20260727.json
```

The profile requires `map` and `odom` to be distinct, forces
`use_sim_time=false`, resolves six distinct topics, and binds the calibration
and physical-site hashes.

## 3. Autonomous Stand Discovery and Facing-Pose Survey

`run_autonomous_stand_exploration.py` is the single-entry real-robot pipeline.
It plans an A* route from the live AMCL pose to a sequence of stopped inspection
poses on one center corridor, fuses LiDAR candidates across the complete
corridor, visits each stable candidate at a 0.7 m pre-approach, and saves the
validated QR-facing pose. If the first camera view observes a stable stand axis
but not the QR, it plans one evidence-bound A* visit to the opposite face and
tries again. That visit starts at the requested 0.7 m standoff; only if the
inflated map blocks that exact pose may it step inward in 0.05 m increments,
never below the profile-derived physical minimum. It fails closed if the
expected candidate count, stand axis,
unique QR identity, clearance, or A* reachability cannot be established.
If a directly measured, content-hashed stand model is available, add
`--stand-model-profile <measured_stand_model.json>`; provisional CAD dimensions
are rejected for operational pose commitment.

The finite yaw on a `stand_discovery_corridor` inspection waypoint is a stopped
endpoint requirement, not a heading constraint for the entire incoming A*
segment. The follower keeps exact-vertex pursuit and the certified route tube
during translation, reaches the inspection position along the sealed segment,
and only then performs the in-place terminal alignment. This matters at the end
of the center corridor, where the only collision-free approach direction may be
opposite the requested inspection yaw. Face-approach route kinds retain their
heading-corridor behavior.

The saved occupancy map is not assumed to contain movable stands. During a
coverage leg the follower first holds a repeated zero command on its normal
front-sector stop, stuck watchdog, or when clearance scaling would reduce a
nonzero linear command below the configured physical motion floor (default
`0.01 m/s`). Adaptive recovery then requires at least three fresh, distinct
post-stop scans whose nearest front returns form one coherent map-frame cluster
while both map and odom show that the robot stayed stationary and their relative
offset stayed stable. One isolated return, including a single `0.234 m` ray, can
never create a keepout. Coherent clear scans resume only on a new complete
safety cycle; missing, inconsistent, moving, or localization-divergent evidence
leaves the robot stopped.

A confirmed blocker becomes a run-local navigation keepout only. It is not a
semantic stand observation and cannot update the stand registry. A new inflated
A* costmap plans back to the same inspection viewpoint. Its exact-start prefix
is either a forward connector inside the controller's translation-heading
envelope or one heading-aligned straight reverse prefix with a separately
clearance-certified reverse-to-forward transition. Only the A* tail may be
line-of-sight simplified. After synchronous planning and sealing, the latest
scan and odom are revalidated and a fresh TF/map pose is looked up; route
admission and the atomic zero-command handoff use that post-plan pose.
Confirmation, geometry, sealing, freshness, join, or A* failure leaves the
robot stopped. Recovery is bounded by `--max-blockage-replans-per-leg` (default
`3`; set `0` to disable it).

The stationary AMCL refresh before preflight and the bounded runtime refresh
are deliberately separate controls. Preflight retains
`--nomotion-update-service /request_nomotion_update` and its 15 s readiness
budget. A runtime stale-`map -> base_footprint` event uses the namespace-relative
`--runtime-nomotion-update-service request_nomotion_update` with at most the
2.0 s configured by `--runtime-nomotion-update-timeout-sec`. This recovery does
not widen `--max-tf-age-sec`: the follower must hold zero, obtain a strictly
newer transform, and pass the normal freshness, command-owner, and certified
route-tube gates before motion may resume. The semantic log records the
configured and namespace-resolved runtime service, both refresh budgets, and
the AMCL-edge future-stamp tolerance.

Physical stand-exploration legs now use an uncertainty-aware split-frame
contract. A* planning, static clearance, transient LiDAR keepouts, and route
certificates remain in `map`. After a stopped AMCL preflight, the runner seals
one direct `map <- odom` transform, projects that exact certified route into
`odom`, and the follower uses only fresh `odom <- base_footprint` poses for
control and route-tube checks. Later AMCL updates are a consistency monitor;
they never steer the robot or rewrite the active route. A correction outside
the covariance allowance already reserved by the route-clearance budget causes
repeated zero and invalidates the active odom certificate. For coverage legs,
the autonomous wrapper may perform one bounded stopped-only recovery by
default: it collects a new stationary AMCL/TF preflight, replans from that
admitted pose to the same committed viewpoint, seals a new map route, runs the
uncertainty admission again, and creates a new odom execution certificate. It
does not widen the correction, route-tube, covariance, or obstacle gates.

The budget samples uninflated static-map clearance and subtracts the robot
radius, collision margin, 30 mm tracking tube, empirical odom drift, braking
distance, the configured localization-sigma allowance, and segment-local yaw
uncertainty. Exact exhaustion rejects. Low spread over five AMCL samples is
evidence of convergence only, not absolute accuracy; physical execution also
requires `--localization-branch-proof-id` naming a known start or asymmetric
landmark. After arrival, LiDAR observation uses exact-time `odom <- base_scan`
composed with the same frozen transform, so a later AMCL correction cannot move
the recorded inspection geometry.

An RViz warning such as `Message Filter dropping message: frame 'base_scan' ...
because the queue is full` is a display-side symptom, not a velocity command or
the direct controller stop cause. It normally means RViz could not transform a
queued scan from `base_scan` into its Fixed Frame at the scan timestamp, or that
the visualization process fell behind. If its timestamps overlap a follower TF
failure, it corroborates a shared TF/publication or host-load problem. Before
the next separately authorized physical run, verify the static LiDAR transform:

```bash
ros2 run tf2_ros tf2_echo base_footprint base_scan
```

Then disable nonessential RViz LaserScan, Path, and history-heavy displays for
the trial. Preserve an external, continuous `/tf` and `/tf_static` capture that
can distinguish `map -> odom` from `odom -> base_footprint`; pre/post snapshots
alone cannot localize an intermittent edge outage. Keep the capture
observe-only and outside the follower's `/cmd_vel` path. A no-motion dry run and
offline tests validate admission and recovery logic only; they are not evidence
that runtime TF recovery or obstacle replanning succeeded on the physical
robot.

Use a new `--session-id` for every command below; sessions are immutable. These
are live-ROS checks, not simulation runs.

First verify the exact runtime graph without authorizing motion:

```bash
ros2 topic echo --once /robot1/amcl_pose
ros2 topic echo --once /robot1/scan
ros2 topic echo --once /robot1/camera/camera_info
ros2 topic info /robot1/cmd_vel -v

python3 scripts/aufgabe04/real_robot/run_autonomous_stand_exploration.py \
  --robot-profile results/aufgabe04/real/profiles/<robot_profile>.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/<physical_site>.json \
  --map maps/aufgabe03/<real_map>.yaml \
  --semantic-map-id <real_map_id> \
  --expected-stand-count 3 \
  --exact-inspection-point-count 2 \
  --session-id stand_explore_dry_001
```

Without `--execute`, the script reads live AMCL, creates the center-corridor
coverage plan, seals the first physical route, and runs the full route/ROS
preflight without publishing velocity. Require
`status=first_leg_dry_run_ok` and `motion_published=false` in
`mission_summary.json`. Inspect the coverage plan, first sealed route,
route diagnostics, and preflight JSON before proceeding.

For the first physical checkpoint, clear the arena, keep the unloaded robot in
view, prepare Ctrl+C plus the physical stop, and keep a second terminal ready
to publish one zero `Twist` to the profile's exact resolved command topic. Then
authorize only one center-corridor leg:

```bash
python3 scripts/aufgabe04/real_robot/run_autonomous_stand_exploration.py \
  --robot-profile results/aufgabe04/real/profiles/<robot_profile>.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/<physical_site>.json \
  --map maps/aufgabe03/<real_map>.yaml \
  --semantic-map-id <real_map_id> \
  --expected-stand-count 3 \
  --max-blockage-replans-per-leg 3 \
  --max-startup-reseals-per-leg 3 \
  --max-runtime-localization-reseals-per-leg 1 \
  --coverage-leg-limit 1 \
  --localization-branch-proof-id <known_start_or_asymmetric_landmark_id> \
  --session-id stand_explore_leg_001 \
  --execute
```

Type `RUN` only after the separate no-motion run has passed and the live
velocity owner is unambiguous. That one mission-level confirmation covers
routine coverage, candidate pre-approach, and opposite-face child legs only
through separate immutable one-use permits. Every child still has to pass its
own dry-run, preflight, route/certificate binding, uncertainty budget, and live
revalidation before atomically claiming its permit immediately before motion;
it does not ask for another `RUN`. A direct standalone child without this
parent-issued contract remains interactive. Startup route reseals remain
outside this scope and require fresh operator confirmation. A
successful checkpoint writes
`status=coverage_leg_checkpoint_complete`, the stopped LiDAR epoch, run events,
preflight evidence, and the next viewpoint ID. If stand recovery occurred,
also inspect `controller_trace.jsonl`, `adaptive_replans.jsonl`,
`coverage/replans/`, and the suffixed
`execution/coverage_leg_<index>_replan_<index>/` certificate bundle. A stop
reason such as route-tube departure, stale TF, or ambiguous velocity ownership
is not classified as a stand and is never auto-replanned. In particular, a
runtime route-tube departure is terminal for that authorization: there is no
in-process recovery or retry. Any continuation requires a separately resealed
route, another no-motion dry-run/preflight, and a new typed `RUN`.

The one exception is an exact coverage-leg global-consistency stop whose
persisted contract contains all of the following: `status=stopped`, prior
motion, `fault_code=localization_reseal_required`,
`source=global_consistency_monitor`, `monitor_action=FORCE_ZERO_RESEAL`,
`fail_closed=true`, and a rejected continuity decision that explicitly
requires both a zero cycle and reseal. The bounded retry count is controlled by
`--max-runtime-localization-reseals-per-leg` (default `1`; set `0` to disable).
The old certificate is never reused. The separate runtime-reseal authorization
derived from the mission-level `RUN` covers only this bounded, same-leg,
same-target recovery class. After the stationary preflight, replacement A*
route, exact-start connector, dry-run, uncertainty budget, and certificate all
pass, the wrapper publishes an immutable one-run motion permit and the child
validates it instead of asking for another `RUN`.
Missing, malformed, already-consumed, scope-mismatched, or artifact-mismatched
permits fail closed before follower motion.

Recovery evidence is written as an ordered sequence in
`adaptive_replans.jsonl`: `runtime_localization_reseal_started`,
`runtime_localization_admitted`, `runtime_localization_route_replanned`, and
`runtime_localization_route_sealed`. The mission authorization and exact child
permit and its atomic one-use receipt are stored under `motion_authorization/`,
and the child semantic log records permit admission. A failed gate instead records
`runtime_localization_reseal_failed` and authorizes no continuation. This
recovery currently applies only to center-corridor coverage legs. Candidate
pre-approach and opposite-face legs remain terminal on the same stop. A
coverage child that already adopted a `transient_navigation_blockage_replanned`
overlay also remains terminal because relaunching it would otherwise reset the
overlay and the per-leg blockage budget.

Immediately after ROS preflight, the runner also binds the fresh
`map -> base_footprint` pose to the first certified route segment before it asks
for motion confirmation. If AMCL has moved outside the unchanged `0.03 m`
startup tube, no velocity is published and the stale certificate is rejected.
For an ordinary coverage leg, the autonomous wrapper may use that rejected
pose to run a complete new A* plan, validate a new exact-start connector, seal
a new certificate, and repeat the dry-run. The previous mission-level `RUN`
does not authorize this replacement: type a fresh `RUN` only after inspecting
the resealed artifact paths printed by the script. The bounded retry count is
controlled by `--max-startup-reseals-per-leg`; dynamic stand-blockage overlays
remain fail-closed rather than being discarded by this generic reseal.

Certified discovery routes also treat every material A* bend as an explicit
control handoff. The follower approaches that vertex to within `0.01 m`, keeps
the incoming segment active while rotating in place toward the outgoing
segment, and publishes a zero-command handoff cycle before translating again.
The in-place hold is limited to `0.025 m`, strictly inside the unchanged
`0.03 m` execution tube; exceeding the hold fails closed.

Next validate the complete center-corridor discovery without approaching any
candidate:

```bash
python3 scripts/aufgabe04/real_robot/run_autonomous_stand_exploration.py \
  --robot-profile results/aufgabe04/real/profiles/<robot_profile>.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/<physical_site>.json \
  --map maps/aufgabe03/<real_map>.yaml \
  --semantic-map-id <real_map_id> \
  --expected-stand-count 3 \
  --exact-inspection-point-count 2 \
  --stop-after-coverage \
  --session-id stand_explore_coverage_001 \
  --execute
```

Require `status=coverage_complete`, the exact expected stand count, a
content-hashed `coverage_candidate_admission.json`, and a content-hashed
`candidate_snapshot.json`. Review the fused candidates in the map frame before
running the complete mission with a fresh session ID:

```bash
python3 scripts/aufgabe04/real_robot/run_autonomous_stand_exploration.py \
  --robot-profile results/aufgabe04/real/profiles/<robot_profile>.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/<physical_site>.json \
  --map maps/aufgabe03/<real_map>.yaml \
  --semantic-map-id <real_map_id> \
  --expected-stand-count 3 \
  --exact-inspection-point-count 2 \
  --session-id stand_explore_full_001 \
  --execute
```

The complete run writes `stand_facing_catalog.json`,
`station_identity_registry.json`, `candidate_snapshot.json`, per-leg evidence
bundles, camera/LiDAR debug artifacts, and `mission_summary.json`. The catalog
authorizes no motion to the final QR-facing poses; it stores those poses for a
later logistics planner. Any failure writes `mission_failure.json` and stops
the mission. Do not add `/behavior_server`, `/velocity_smoother`, or any other
unexpected `/cmd_vel` publisher to an allowlist; resolve ownership instead.

## 4. Prepare and Collect a Passive Survey

Use the frozen map, detector-produced candidate snapshot, and complete station
identity registry from the successful sealed workflow:

```bash
python3 scripts/aufgabe04/real_robot/prepare_passive_survey.py \
  --robot-profile results/aufgabe04/real/profiles/robot1_unloaded_20260727.json \
  --camera-calibration results/aufgabe04/real/profiles/robot1_camera_20260727.json \
  --physical-site docs/setups/aufgabe04_lab_v1.json \
  --map maps/aufgabe03/<real_map>.yaml \
  --semantic-map-id <real_map_id> \
  --candidate-snapshot results/aufgabe04/<candidate_snapshot>.json \
  --station-identity-registry results/aufgabe04/<identity_registry>.json \
  --output-dir results/aufgabe04/real/surveys/real_survey_001 \
  --catalog results/aufgabe04/real/surveys/real_survey_001/catalog.json \
  --catalog-id real_survey_001 \
  --session-id real_survey_001 \
  --survey-manifest results/aufgabe04/real/surveys/real_survey_001/manifest.json
```

The command prints the immutable plan path. For each `candidate_runs` entry:

1. Manually place the stopped robot at a clear, localized viewpoint.
2. Run its `observer_command`.
3. Inspect `observer_status.json` and the debug images.
4. Run its `catalog_validation_command` only after the expected QR ID and
   consensus evidence were committed.

The observer creates no ROS publisher. It rejects simulated time, stale or
unsynchronized sensors, changed `CameraInfo`, changed camera extrinsics,
non-stationary poses, missing exact-time TF, projected-target/LiDAR
disagreement, weak silhouette consensus, and incorrect QR identity. A raw
compressed image is rectified into the sealed `CameraInfo.p` geometry before
the projected ROI is evaluated.

The real-camera stand-axis settings are an **offline candidate profile**, not a
hardware-validated detector profile. Its default preprocessing is
`--edge-preprocess channel-union`, which preserves color-channel boundaries
that can disappear in grayscale. `--edge-preprocess gray` remains available
for a controlled comparison, and the existing `--canny-low`/`--canny-high`
overrides remain bounded to `0 <= low < high <= 255`. The projected head size
resolves the minimum contour area, minimum side height, and conservative
odd-valued close kernel; the current broad square-face aspect gate remains
`0.45..1.80`.

At each image timestamp, the exact `camera_optical <- map` TF transforms the
known world-vertical top and bottom of the stand head. Live `CameraInfo.p`
projects that 3D line into the rectified image, including camera roll, and the
resulting direction is passed into the same silhouette estimator used by the
existing façade. There is no additional temporal smoother: only the current
raw usable estimate may enter `AxisConsensusAccumulator`.

When `perception_debug/` is enabled, the observer refreshes:

- `latest_frame.png` and `latest_head_roi.png`
- `latest_edges.png` (topology edges)
- `latest_raw_edges.png` (untouched measurement edges, when available)
- `latest_side_evidence.png` (selected raw side support, when available)
- `latest_rectangle_mask.png` and `latest_rectangle_overlay.png` (when
  available)
- `latest_metadata.json` with the resolved profile, projected side direction,
  estimator status, and the exact artifact list

Unavailable optional images are removed instead of leaving stale
`latest_*.png` evidence from an older frame.

Before describing this profile as real-camera validated, collect representative
hardware captures across stand colors, QR texture, lighting, distance, camera
pitch/roll, and background clutter. Compare the default and grayscale modes on
those frozen captures, record false positives and unavailable estimates, and
measure rectification plus estimator latency and dropped/stale-frame behavior
at the intended onboard processing rate. Those capture and latency results are
required evidence; passing the offline tests below is not a hardware claim.

After all expected candidates are resolved, run the plan's
`finalize_command`. Finalization refuses incomplete catalogs or provenance that
differs from the sealed map, snapshot, identity registry, site, profile,
calibration, or survey configuration.

## 5. Unloaded Single-Leg Validation

`run_unloaded_segment.py` accepts the full certified mission artifact chain and
derives namespace, topics, frames, footprint, and speed limits only from the
sealed profile. Its default is a dry run:

```bash
python3 scripts/aufgabe04/real_robot/run_unloaded_segment.py \
  --robot-profile <robot_profile.json> \
  --physical-site <physical_site.json> \
  --route-csv <route.csv> \
  --diagnostics-json <route_diagnostics.json> \
  --route-certificate-json <route_certificate.json> \
  --route-bundle-json <route_bundle.json> \
  --planner-config-json <planner_config.json> \
  --mission-plan-manifest <mission_manifest.json> \
  --survey-manifest <real_survey_manifest.json> \
  --runtime-map-bundle-json <runtime_map_bundle.json> \
  --candidate-snapshot <candidate_snapshot.json> \
  --station-identity-registry <identity_registry.json> \
  --arrival-pose-catalog <real_catalog.json> \
  --task-snapshot <fresh_task_snapshot.json> \
  --leg-index 0 \
  --run-id real_unloaded_leg_001 \
  --confirm-unloaded
```

Before any physical execution, clear the arena, keep an operator beside the
robot with Ctrl+C and physical stop access, and keep a separate terminal ready
to publish one zero `Twist` to the exact resolved command topic. Verify one
localization owner and exclusive velocity ownership. Only then repeat the same
command with `--execute`; it is automatically evidence-bundled and the existing
inner runner still performs ROS preflight and requires typed `RUN`.

This adapter intentionally excludes cargo, loaded footprint, multi-leg
dispatch, strict post-arrival QR events, and fleet coordination.

## Validation

The migration modules remain ROS-free at import and test time:

```bash
python3 -m unittest tests.aufgabe04.test_real_robot_pipeline
python3 -m unittest discover -s tests/aufgabe04
python3 -m compileall -q scripts/aufgabe04/real_robot
```

Passing these checks validates contracts and command construction, not physical
calibration accuracy, localization quality, stopping distance, or route
clearance on the robot.
