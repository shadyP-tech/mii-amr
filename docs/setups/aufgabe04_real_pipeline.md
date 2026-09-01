# Aufgabe 04 Real-Robot Experiment Pipeline

The real experiment path reuses the simulation pipeline's immutable station,
catalog, route, and mission contracts, but replaces simulated sensing and
runtime assumptions with dedicated hardware adapters. It is not approval for a
loaded logistics mission or a two-robot run; see
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md).

## Module Boundary

| Module | Responsibility | Motion capability |
| --- | --- | --- |
| `real_robot/entrypoints/capture_camera_calibration.py` | Capture live `CameraInfo` and the measured `base <- camera_optical` TF | None |
| `real_robot/entrypoints/create_hardware_profile.py` | Seal topics, namespace, frames, site, calibration, footprint, and speed limits | None |
| `perception/stand_axis/model_profile.py` | Load content-hashed geometry and enforce the measured-physical operational contract | None |
| `perception/stand_axis/model_pipeline.py` | Acquire/track the stand pose, project the model, and require current-frame rail refinement | None |
| `perception/stand_axis/model_projection.py` | Project measured head, QR, depth, and stem landmarks into the rectified image | None |
| `perception/stand_axis/model_refinement.py` | Validate current-frame Canny support inside narrow model-projected corridors | None |
| `perception/stand_axis/pose_tracking.py` | Retain a short-lived, same-model/same-camera pose hint between frames | None |
| `perception/stand_axis/real_camera_profile.py` | Validate and resolve the operational model-refinement edge recipe | None |
| `navigation/runtime_localization_reseal.py` | Classify the exact global-consistency zero/reseal contract and enforce its retry budget | None |
| `navigation/runtime_motion_authorization.py` | Bind the mission-level `RUN` to one exact, same-target runtime-localization recovery child and its fresh artifacts | None |
| `navigation/runtime_motion_consumption.py` | Atomically consume that exact child permit once and reject replay before follower motion | None |
| `navigation/localization/startup_active_localization.py` | Validate the bounded rotation policy and content-hashed `LOCALIZE` evidence without ROS | None |
| `navigation/mission_leg_motion_permit.py` | Bind one mission-level `RUN` to separately sealed routine coverage, candidate, and opposite-face child legs | None |
| `navigation/mission_leg_motion_consumption.py` | Atomically consume each exact routine-leg permit once immediately before motion | None |
| `navigation/startup_reseal_motion_authorization.py` | Bind the mission-level `RUN` to an exact bounded same-target pre-motion recovery and its fresh artifacts | None |
| `navigation/startup_reseal_motion_consumption.py` | Atomically consume each startup-reseal permit once immediately before motion | None |
| `navigation/spatial_assignment.py` | Globally associate stopped-epoch detections with existing candidates by cardinality and total distance | None |
| `navigation/coverage_candidate_admission.py` | Fail closed between LiDAR coverage and candidate approaches unless coverage and multi-view candidate evidence are complete | None |
| `navigation/coverage_candidate_lifecycle.py` | Classify LiDAR, multi-view, camera-queue, confirmed, and rejected evidence; evaluate exact-two LiDAR checkpoint completion without constructing a motion snapshot | None |
| `navigation/exact_two_camera_admission.py` | Build and verify the content-hashed exact-two LiDAR-to-camera population handoff without promoting registry lifecycle state | None |
| `real_robot/mission/modes.py` | Resolve one explicit workflow mode and authorization scope; reject contradictory legacy flags, unsafe session IDs, and misleading session labels | None |
| `real_robot/execution/artifact_paths.py` | Canonicalize existing sealed child artifacts so dry evidence, permits, and live argv bind one filesystem identity | None |
| `real_robot/execution/child_runner.py` | Build child-runner and bundle argv, and parse one unambiguous append-only terminal outcome | None |
| `real_robot/readiness/localization.py` | Classify the one bounded no-motion retryable uncertainty-admission failure without changing any limit | None |
| `real_robot/readiness/active_localization.py` | Catch only a typed initial route-uncertainty rejection, require a bounded localization child, admit fresh stopped AMCL, and retry planning | None; injected effects only |
| `real_robot/execution/startup_active_localization.py` | Bind one child process to the rejected selection, `LOCALIZE` authorization, controller trace, stopped-odom proof, and immutable result | Delegates only to the sole waypoint-follower motion edge |
| `real_robot/mission/session_manifest.py` | Snapshot and content-hash resumable coverage checkpoints and terminal survey evidence; manifests explicitly authorize no motion | None |
| `real_robot/mission/checkpoint_resume.py` | Re-hash, restore, and freshly replan one next coverage leg in a new session | None |
| `real_robot/coverage_leg/replanning.py` | Rebuild a coverage leg from admitted startup/runtime-localization evidence while preserving bounded transient-overlay continuity | None; offline route/artifact reconstruction only |
| `real_robot/readiness/startup_reseal.py` | Adapt the startup-reseal safety contract to autonomous coverage execution | None; ROS-free permit construction only |
| `real_robot/configuration/site_contract.py` | Bind site, map bytes, robot profile, and canonical stand count before planning | None |
| `real_robot/coverage_leg/execution.py` | Run the bounded per-leg coverage retry/reseal state machine behind injected ROS and child-process effects | None; delegates any authorized motion to the existing child runner |
| `real_robot/mission/coverage.py` | Commit each completed coverage leg as one ordered observe/fuse/checkpoint transaction and gate candidate materialization | None; cannot execute a leg itself |
| `real_robot/candidate/approach.py` | Order frozen candidates, orchestrate sealed pre-approach/opposite-face inspection, and publish validated identity/facing artifacts behind injected live effects | None; cannot sample ROS, prompt, launch a process, or publish motion itself |
| `real_robot/observer/node.py` | Synchronize image, scan, and exact-time TF; rectify the image; validate measured-model, LiDAR, and QR evidence | None |
| `real_robot/entrypoints/run_autonomous_stand_exploration.py` | Wire CLI/profile/operator authorization to the focused coverage, child-runner, and inspection modules | Dry-run by default; explicit physical gate |
| `real_robot/passive_survey/prepare.py` | Produce immutable per-candidate observer and catalog-validation commands | None |
| `real_robot/passive_survey/finalize.py` | Freeze a complete real arrival catalog and write a real `SurveyManifest` | None |
| `real_robot/entrypoints/run_unloaded_segment.py` | Bind one certified route leg to the sealed hardware/site profile and existing preflight runner | Dry-run by default; explicit physical gate |

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

Keep the orchestration split explicit. Run planning, checkpointing, diagnostic
bundles, camera processing, and all session artifact writes on the workstation
inside its ROS 2 Humble Apptainer environment. The TurtleBot host should run
only the required sensor/actuator bringup nodes. Before every session, verify
the robot's free filesystem space and do not copy capture-heavy session roots
or bags to it:

```bash
ssh turtle 'hostname; df -h /; free -h; printenv ROS_DOMAIN_ID TURTLEBOT3_MODEL LDS_MODEL'
```

The workstation login shell is not itself proof of a usable ROS graph. Enter
the same Apptainer image used for the run, source ROS Humble and the workspace,
then verify the resolved profile topics and frames before the typed `RUN` is
shown. In particular require fresh `/scan`, `/odom`, `/tf`, `/amcl_pose`, and
camera data plus unambiguous ownership of the exact resolved `/cmd_vel` topic.
An empty graph, missing bringup, a mismatched DDS domain/model, or insufficient
robot storage is a stop condition, not a reason to weaken a route or
localization limit.

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
python3 scripts/aufgabe04/real_robot/entrypoints/capture_camera_calibration.py \
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
python3 scripts/aufgabe04/real_robot/entrypoints/create_hardware_profile.py \
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

`entrypoints/run_autonomous_stand_exploration.py` is the single-entry real-robot pipeline.
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
Both `execute-exact-two-camera` and `execute-full` require
`--stand-model-profile <measured_physical_stand_model.json>`. The profile must
be content-hashed, declare `environment=physical`, and declare
`measurement_status=measured`, and use the complete schema-v2 geometry.
Missing, legacy-v1, provisional, or simulation geometry is rejected before the
session directory, planning, typed `RUN`, or any coverage motion. The checked-in
operational profile is
`configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json`.
It records the 78 mm square by 6 mm deep head, 210 mm floor-to-head-top
(171 mm derived head centre), 153 mm square maximum base footprint, 71 mm
paper panel, and a 62 mm photo-rectified QR symbol boundary with 2 mm model
tolerance. The older `physical_stand_assumptions_v1.json` remains diagnostic
and cannot authorize a real run.

The base and LiDAR geometries are deliberately separate. The base contributes
a conservative floor-level collision radius of about `0.110187 m`; the
coverage/observer candidate radius remains the established `0.06 m` LiDAR
envelope. Replacing the latter with the base radius would widen target
association and break frozen-candidate binding. Candidate uncertainty is
charged once to both clearance paths. With the current TurtleBot profile and
runtime margins, the model-derived active minimum is `0.33 m`; the existing
`0.35 m` final-facing offset remains admissible and is the minimum reviewed
operational value for this profile.

The entrypoint is only the live-edge adapter. Coverage route reconstruction is
isolated in `coverage_leg/replanning.py`; the bounded per-leg state
machine remains in `coverage_leg/execution.py`; and candidate ordering,
opposite-face fallback, facing validation, and identity/catalog publication are
owned by `candidate/approach.py`. The extracted modules cannot read
ROS, prompt, launch subprocesses, or publish velocity. Fresh AMCL reads, passive
camera capture, exact one-use mission-leg permits, and child motion remain
explicit injected effects supplied by the entrypoint.

The finite yaw on a `stand_discovery_corridor` inspection waypoint or
`detected_stand_preapproach` camera standoff is a stopped endpoint requirement,
not a heading constraint for the entire incoming A* segment. The follower keeps
exact-vertex pursuit and the certified route tube during translation, reaches
the inspection position along the sealed segment, and only then performs the
in-place terminal alignment. This matters when the only collision-free approach
direction may be opposite the requested inspection yaw. Face-approach route
kinds retain their heading-corridor behavior.

If an authorized candidate startup-reseal child publishes motion and then
stops, the coordinator remains fail-closed and does not start another reseal or
camera capture. Its terminal event and `mission_failure.json` preserve the
child run ID, `stop_reason`, structured stop details, motion state, and consumed
permit evidence so the actionable controller failure is not hidden by the
recovery-policy boundary.

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

If a dry child fails only because this unchanged route-specific uncertainty
budget is exhausted, the wrapper may request a fresh no-motion AMCL update and
repeat the complete dry admission with a new child ID. This is bounded by
`--max-localization-readiness-retries-per-leg` (default `2`; set `0` to
disable). It does not mint a permit, lower the sigma multiplier, shrink the
collision allowance, or widen the `0.03 m` route tube. Any other preflight
failure remains terminal.

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

When these commands are wrapped by `scripts/common/run_with_bundle.sh`, its
independent topic, node, action, sensor, status, and TF captures start in
parallel and are all joined before the child command starts. No diagnostic or
timeout is skipped; the latency reduction is evidence-preserving and does not
replace the child's safety preflight.

First verify the exact runtime graph without authorizing motion:

```bash
ros2 topic echo --once /robot1/amcl_pose
ros2 topic echo --once /robot1/scan
ros2 topic echo --once /robot1/camera/camera_info
ros2 topic info /robot1/cmd_vel -v

python3 scripts/aufgabe04/real_robot/entrypoints/run_autonomous_stand_exploration.py \
  --robot-profile configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/aufgabe04_lab_20260817.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --expected-stand-count 5 \
  --run-mode dry-first-leg \
  --session-id stand_explore_dry_001
```

In `dry-first-leg` mode, the script reads live AMCL, creates the center-corridor
coverage plan, seals the first physical route, and runs the full route/ROS
preflight without publishing velocity. Require
`status=first_leg_dry_run_ok` and `motion_published=false` in
`mission_summary.json`. Inspect the coverage plan, first sealed route,
route diagnostics, and preflight JSON before proceeding.

All autonomous execute modes now perform the same first-route readiness phase
before displaying the typed `RUN` prompt. With `--prompt-for-initialpose`, the
script first passively proves that one fresh `LaserScan` has an exact-time
`odom <- base_scan` transform, then pauses once before preplanning localization
admission. Click RViz **2D Pose Estimate** only at that prompt while the robot
is stopped. The phase admits the post-click AMCL pose, creates the
center-corridor coverage plan from that pose, seals a separate nonauthorizing
copy of the first route, and runs its dry uncertainty admission. If the AMCL
uncertainty budget is temporarily exhausted after the route is sealed, the
script prints a no-motion retry message and repeats AMCL admission without
another RViz click. Do not compensate by raising the uncertainty limit,
shrinking the sigma multiplier, or widening the certified route tube. A
rejection writes
`preflight/lidar_scan_tf_before_authorization.json` and
`authorization_readiness/coverage_leg_<index>/readiness.json`, exits without
requesting `RUN`, and issues no motion authorization or permit. If the bounded
budget expires, leave the robot stopped, correct the pose, and start a fresh
session ID; failed session directories remain immutable evidence.

For the exact-two physical workflow, startup active localization is an
explicit opt-in fallback for a narrower condition: the first stopped AMCL
admission has already passed, but the typed startup route selector rejects
every reachable initial route on its unchanged uncertainty budget. Enable it
with `--enable-startup-active-localization`. The parent then requests a
separate typed `LOCALIZE`, never `RUN`, for one odometry-measured in-place
rotation. The default is one near-full revolution at `0.12 rad/s`, bounded by
`70 s`; translation commands are forbidden, measured translation above
`0.03 m` stops the phase, and fresh LiDAR, odometry, a `0.20 m` scan envelope,
exclusive `/cmd_vel` ownership, repeated zero commands, and a stopped odometry
pair are mandatory. Afterward, the parent performs a fresh stationary AMCL/TF
admission and retries initial planning with new attempt-specific evidence.
Only a new successful plan can reach the later, separate typed `RUN` prompt.

The phase does not recover a failed initial AMCL preflight, a generic planner
error, an obstacle stop, or any failure after mission motion. It never changes
the `0.03 m` certified route tube, uncertainty sigma multiplier, collision
margin, or route-clearance admission. Its evidence is stored under
`startup_active_localization/attempt_<index>/`, including the rejected route
selection, content-hashed preflight/authorization/result, controller trace,
semantic events, and post-motion preplanning localization receipt. For a
reviewed experiment, use the explicit defaults (or tighter previously
validated values):

```bash
  --enable-startup-active-localization \
  --max-startup-active-localization-attempts 1 \
  --startup-active-localization-rotation-rad 6.283185307179586 \
  --startup-active-localization-angular-speed-radps 0.12 \
  --startup-active-localization-timeout-sec 70
```

This readiness receipt is advisory, not a motion permit. After the operator
types `RUN`, the normal execution path independently seals the route again and
repeats the dry preflight, uncertainty budget, live checks, and one-use permit
claim. That second chain remains authoritative if AMCL, TF, route bytes, or ROS
ownership changes while the operator reviews the prompt.

For the first physical checkpoint, clear the arena, keep the unloaded robot in
view, prepare Ctrl+C plus the physical stop, and keep a second terminal ready
to publish one zero `Twist` to the profile's exact resolved command topic. Then
authorize only one center-corridor leg:

```bash
python3 scripts/aufgabe04/real_robot/entrypoints/run_autonomous_stand_exploration.py \
  --robot-profile configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/aufgabe04_lab_20260817.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --expected-stand-count 5 \
  --max-blockage-replans-per-leg 3 \
  --max-startup-reseals-per-leg 3 \
  --max-runtime-localization-reseals-per-leg 1 \
  --max-localization-readiness-retries-per-leg 2 \
  --run-mode execute-coverage-checkpoint \
  --coverage-leg-limit 1 \
  --localization-branch-proof-id <known_start_or_asymmetric_landmark_id> \
  --session-id stand_explore_checkpoint_001
```

The physical-site descriptor is the canonical source of the five-stand count.
`--expected-stand-count 5` is an optional assertion: omitting it derives five,
while any other value fails before planning, session creation, or a `RUN`
prompt. The robust `execute-full` workflow lets stop spacing determine the
redundant centerline viewpoint set and still rejects
`--exact-inspection-point-count 2`.

For a bounded two-stop LiDAR check, combine
`--exact-inspection-point-count 2`, `--coverage-leg-limit 2`, and
`--run-mode execute-coverage-checkpoint`. Successful completion means exactly
five active static-map-admitted LiDAR candidates passed the frozen count and
basic evidence gate. The result is a terminal, non-resumable checkpoint with
`camera_approach_authorized=false`; it does not promote single-view candidates
to `pending_camera`, create a candidate snapshot, or continue into camera
approach motion. Exact-two planning samples longitudinal center-corridor
candidates at the reviewed `0.40 m` density and requires a persisted `1.00 m`
world-space baseline, while retaining the `95%` coverage and shared-visibility
gates.

For the explicit two-stop-to-camera workflow, use
`--run-mode execute-exact-two-camera` with
`--exact-inspection-point-count 2`; omit `--coverage-leg-limit` or set it to
exactly `2`. This mode first seals terminal LiDAR evidence, then constructs a
content-hashed handoff for exactly five active static-map-admitted candidates.
The handoff preserves which candidates are multi-view `pending_camera` and
which are single-view `provisional`; only its bound camera decision path may
resolve the latter. It continues in the same process under the initial `RUN`,
while every candidate and opposite-face motion still requires its own sealed
route, dry-run, live gates, and atomically consumed one-use permit.
Before each candidate route is planned, a stopped AMCL sample window must be
paired with direct dynamic `map <- odom` samples and followed by a fresh,
consistent transform lookup; missing or drifting evidence stops before route
planning.

If one of those camera routines stops after motion with the exact global
localization-consistency `FORCE_ZERO_RESEAL` contract, the parent may use the
independent `--max-runtime-localization-reseals-per-leg` budget. It first
admits fresh stationary AMCL/TF evidence, replans the same routine and target,
repeats the dry/live route gates, and issues a new one-use runtime permit. The
initial mission `RUN` still applies; target changes, generic safety stops,
malformed evidence, stale localization, permit reuse, and exhausted budgets
remain terminal.

Type `RUN` only after the automatic preauthorization readiness report has
passed (and, when using a staged protocol, after reviewing the separate
`dry-first-leg` session) and the live velocity owner is unambiguous. In this
`execute-coverage-checkpoint` mode that one mission-level
confirmation covers coverage child legs only, through separate immutable
one-use permits. Candidate and opposite-face leg kinds are absent from the
master authorization. Every child still has to pass its
own dry-run, preflight, route/certificate binding, uncertainty budget, and live
revalidation before atomically claiming its permit immediately before motion;
it does not ask for another `RUN`. A direct standalone child without this
parent-issued contract remains interactive. An eligible certified-start
mismatch or exact zero-motion prestart `map <- odom` consistency stop also
does not ask again: it first admits fresh stationary AMCL/TF, reconstructs the
same target, repeats the dry/live gates, and then uses a separate bounded
startup-reseal authorization and exact one-use recovery permit. Missing,
tampered, stale, target-changed, over-budget, or replayed evidence stops instead
of prompting or moving. A successful checkpoint writes
`status=coverage_leg_checkpoint_complete`, the stopped LiDAR epoch, run events,
preflight evidence, the next viewpoint ID, and an immutable content-hashed
`checkpoints/coverage_leg_<count>/manifest.json`. The manifest snapshots the
plan, progress, survey summary, registry, and latest LiDAR observer summary;
it explicitly records `motion_authorized=false`. The mission summary keeps
`motion_published=true` as historical evidence that the completed leg moved,
and repeats that fact as `prior_leg_motion_published=true`; neither field is a
permit for continuation. Require `next_required_action=resume-next-coverage-leg`
and use a new session with fresh localization. If stand recovery occurred,
also inspect `controller_trace.jsonl`, `adaptive_replans.jsonl`,
`coverage/replans/`, and the suffixed
`execution/coverage_leg_<index>_replan_<index>/` certificate bundle. A stop
reason such as route-tube departure, stale TF, or ambiguous velocity ownership
is not classified as a stand and is never auto-replanned. In particular, a
runtime route-tube departure is terminal for that authorization: there is no
in-process recovery or retry. Any continuation requires a separately resealed
route, another no-motion dry-run/preflight, and a new typed `RUN`.

The exact-time LiDAR TF check is repeated immediately before every coverage
motion and again before starting each LiDAR observer epoch. A missing or stale
`odom <- base_scan` chain therefore stops before a leg permit/live child or
before the eight-second observer, respectively. If this gate reports
`exact_time_transform_unavailable`, restore the TurtleBot bringup/static
transform publisher and verify `base_footprint -> base_scan`; do not add a
second velocity publisher or bypass the gate.

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
overlay carries a content-hashed resume-state binding through the same-target
localization reseal; the adopted overlay, source run IDs, route hashes, and
remaining blockage budget must all verify before a new permit can be minted.
Missing or mismatched overlay evidence remains terminal.

Each independently sealed route CSV has its own local artifact leg index,
normally `0`; coverage/replan and mission indices remain separate evidence
identities and never select a CSV row. After the no-motion dry child has
created its evidence, the parent canonicalizes the session, route,
diagnostics, certificate, and authorization paths before hashing, permit
publication, and live child argv construction. This prevents relative-path or
platform-alias differences such as macOS `/var` versus `/private/var` from
invalidating an otherwise exact permit.

Immediately after ROS preflight, the runner also binds the fresh
`map -> base_footprint` pose to the first certified route segment before it asks
for motion confirmation. If AMCL has moved outside the unchanged `0.03 m`
startup tube, no velocity is published and the stale certificate is rejected.
The follower's bounded initial sensor wait also holds zero until its own fresh
TF buffer can validate both the odom-owned control pose and the read-only
`map -> odom` global-consistency edge. A cold missing edge may populate inside
that existing wait. Persistent missing, stale, or future TF and exact
translation/yaw drift stop the child before a nonzero command and may enter
the same bounded fresh-localization/reseal path; malformed, conflicting, or
unknown evidence remains terminal. For an ordinary coverage leg, the
autonomous wrapper samples a fresh stationary pose, runs a complete
same-target A* plan, validates a new exact-start connector, seals a new
certificate, and repeats the dry-run. The original mission-level `RUN` covers
the replacement only through the dedicated startup master plus an exact permit
binding the rejected no-motion log, recovery source, fresh localization,
replacement route/certificate, and dry artifacts. The child claims that
permit once after every live gate and immediately before the semantic
`motion_started` execution-attempt marker; that marker precedes the follower
and is not evidence of a nonzero Twist. No second prompt occurs. The bounded
retry count is controlled by `--max-startup-reseals-per-leg`; any adopted
dynamic blockage overlay must be preserved and rebound rather than discarded
by this reseal.

To continue exactly one remaining coverage leg from a successful checkpoint,
use a new session ID and repeat every behavior-relevant option from the parent
run:

```bash
python3 scripts/aufgabe04/real_robot/entrypoints/run_autonomous_stand_exploration.py \
  --robot-profile configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/aufgabe04_lab_20260817.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --expected-stand-count 5 \
  --run-mode resume-next-coverage-leg \
  --resume-checkpoint results/aufgabe04/real/autonomous_exploration/<parent_session>/checkpoints/coverage_leg_001/manifest.json \
  --localization-branch-proof-id <fresh_known_start_or_landmark_id> \
  --session-id stand_explore_resume_002
```

The checkpoint and every referenced snapshot are re-hashed before ROS
preflight. The new session then obtains fresh stationary AMCL/TF evidence and
plans a new exact-start A* route to the committed next viewpoint. Old routes,
semantic logs, authorizations, and permits are never reused. The new typed
`RUN` authorizes that one coverage leg only.

Certified discovery routes also treat every material A* bend as an explicit
control handoff. The follower approaches that vertex to within `0.01 m`, keeps
the incoming segment active while rotating in place toward the outgoing
segment, and publishes a zero-command handoff cycle before translating again.
The in-place hold is limited to `0.025 m`, strictly inside the unchanged
`0.03 m` execution tube; exceeding the hold fails closed.

Next validate the complete center-corridor discovery without approaching any
candidate:

```bash
python3 scripts/aufgabe04/real_robot/entrypoints/run_autonomous_stand_exploration.py \
  --robot-profile configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/aufgabe04_lab_20260817.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --expected-stand-count 5 \
  --run-mode execute-coverage-only \
  --localization-branch-proof-id <known_start_or_asymmetric_landmark_id> \
  --session-id stand_explore_coverage_001
```

After every stopped LiDAR epoch, the wrapper samples fresh stationary
localization and seals the next leg from that pose instead of reusing the pose
from before the observation wait. Candidate fusion performs one global
maximum-cardinality, minimum-total-distance assignment per epoch, so input
order cannot greedily create avoidable provisional duplicates. Neither change
widens the existing candidate merge-distance gate.

Require `status=coverage_complete`, the exact expected stand count, a
content-hashed `coverage_candidate_admission.json`, and a content-hashed
`candidate_snapshot.json`. Review the fused candidates in the map frame before
running the complete mission with a fresh session ID:

```bash
python3 scripts/aufgabe04/real_robot/entrypoints/run_autonomous_stand_exploration.py \
  --robot-profile configs/aufgabe04/real_robot_profiles/turtlebot1_unloaded_20260817.json \
  --camera-calibration results/aufgabe04/real/profiles/<camera_calibration>.json \
  --physical-site docs/setups/aufgabe04_lab_20260817.json \
  --map maps/aufgabe03/arena_1p898x3p9_auto.yaml \
  --semantic-map-id arena_1p898x3p9_auto \
  --expected-stand-count 5 \
  --stand-model-profile configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json \
  --final-facing-offset-m 0.35 \
  --run-mode execute-full \
  --localization-branch-proof-id <known_start_or_asymmetric_landmark_id> \
  --session-id stand_explore_full_001
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
python3 scripts/aufgabe04/real_robot/entrypoints/prepare_passive_survey.py \
  --robot-profile results/aufgabe04/real/profiles/robot1_unloaded_20260727.json \
  --camera-calibration results/aufgabe04/real/profiles/robot1_camera_20260727.json \
  --stand-model-profile configs/aufgabe04/stand_models/physical_stand_measured_20260826_v2.json \
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

The passive survey's `0.33 m` target is observation-only and has no motion
authority. It must not be promoted directly to a robot route; any later motion
must pass the autonomous model-derived active-stand clearance and facing-route
validation against fresh localization and map evidence.

The observer creates no ROS publisher. It rejects simulated time, stale or
unsynchronized sensors, changed `CameraInfo`, changed camera extrinsics,
non-stationary poses, missing exact-time TF, projected-target/LiDAR
disagreement, weak model-refinement consensus, and incorrect QR identity. A raw
compressed image is rectified into the sealed `CameraInfo.p` geometry before
the projected ROI is evaluated.

The real-camera stand-axis settings control **model-refinement edge evidence**,
not a free-form silhouette detector. The default preprocessing is
`--edge-preprocess channel-union`, which preserves color-channel boundaries
that can disappear in grayscale. `--edge-preprocess gray` remains available
for a controlled comparison, and `--canny-low`/`--canny-high` remain bounded
to `0 <= low < high <= 255`. Those edges can support only narrow corridors
projected from the measured stand model; they cannot propose an unrelated
stand shape.

At each image timestamp, the exact `camera_optical <- map` TF and live
`CameraInfo.p` project a conservative target ROI from the mapped stand center.
Inside that ROI, QR geometry or a short-lived same-profile pose hint seeds the
measured model projection. A frame contributes to consensus only when the
projected head rails are supported and re-fitted by the current frame. The
operational observer never calls the historical free-form silhouette detector
and never commits a predicted-only model pose.

When `perception_debug/` is enabled, the observer refreshes:

- `latest_frame.png` and `latest_head_roi.png`
- `latest_edges.png` and `latest_raw_edges.png` (current-frame model-refinement edges)
- `latest_side_evidence.png` (current-frame support inside projected corridors)
- `latest_rectangle_mask.png` and `latest_rectangle_overlay.png` (when
  available)
- `latest_metadata.json` with `estimator_mode=metric_model_only`, measured-model
  ID/hash/status, refinement state, estimator status, and the exact artifact list

Unavailable optional images are removed instead of leaving stale
`latest_*.png` evidence from an older frame.

Before describing a measured profile as real-camera validated, collect
representative hardware captures across stand colors, QR texture, lighting,
distance, camera pitch/roll, and background clutter. Record model acquisition,
current-frame refinement, ambiguity, false positives, unavailable estimates,
and per-stage latency at the intended processing rate. Passing offline tests
alone is not a hardware claim.

After all expected candidates are resolved, run the plan's
`finalize_command`. Finalization refuses incomplete catalogs or provenance that
differs from the sealed map, snapshot, identity registry, site, profile,
calibration, or survey configuration.

## 5. Unloaded Single-Leg Validation

`run_unloaded_segment.py` accepts the full certified mission artifact chain and
derives namespace, topics, frames, footprint, and speed limits only from the
sealed profile. Its default is a dry run:

```bash
python3 scripts/aufgabe04/real_robot/entrypoints/run_unloaded_segment.py \
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
