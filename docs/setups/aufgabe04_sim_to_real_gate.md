# Aufgabe 04 Simulation-to-Real Migration Gate

**Current decision: not cleared for real-world mission execution.** The
simulation pipeline now has fail-closed, ROS-free provenance and safety
contracts. A dedicated passive real sensing/profile path and an unloaded
single-leg wrapper are implemented, but they do not yet have hardware evidence
and several active runtime contracts remain missing. Passing unit tests or
completing a selected Gazebo route does not clear this gate. Use
[`aufgabe04_real_pipeline.md`](aufgabe04_real_pipeline.md) for the new adapter
boundary and evidence workflow.

## Sealed Artifact Chain

Use a new content-derived filename or run directory whenever an input changes.
The artifact store permits an identical retry but refuses to overwrite an
existing path with different content.

```text
map YAML + map image ──> frozen map bundle
                            │
LiDAR confirmations ──> candidate snapshot ──> station identity registry
                            │                         │
simulation or passive-real survey ──> frozen arrival catalog ──> survey manifest
                                                      │
fresh server data ──> validated task snapshot ────────┤
                                                      v
                              task-ordered route + route certificate
                                                      │
                                             mission-plan manifest
                                                      │
                                             execution evidence manifest
```

The implemented bindings are:

- A frozen map bundle hashes the YAML bytes, referenced image bytes, parsed map
  geometry, semantic map ID, and planning frame.
- A candidate snapshot records the complete candidate set, geometry,
  uncertainty, keepout, detector/source hashes, observation IDs, and parent map
  bundle hash. It is generated from confirmed detector observations by
  `plan_detected_stand_exploration.py`.
- Each detector observation now carries the full YAML+image map-bundle digest;
  the candidate planner rejects observations from another image even when the
  YAML bytes and frame name are unchanged. Schema-v2 observations also bind
  the LaserScan stamp to an exact-time map-to-scan TF query and persist the ROS
  clock, scan age, TF age, and TF/scan skew for downstream recomputation.
- The one-to-one identity registry binds every candidate UID to exactly one QR
  ID and server station ID and is itself bound to the candidate snapshot.
- A survey manifest links the map bundle, candidate snapshot, environment
  descriptor, survey configuration, calibration profile, and completed arrival
  catalog. `run_arrival_pose_survey.py` emits the simulation form;
  `finalize_passive_survey.py` emits the real form bound to a physical-site
  descriptor. The legacy candidate JSON escape hatch is not migration evidence.
- A validated task snapshot binds the fresh status/plan timestamps, source
  plan hash, robot and mission IDs, and exact remaining server station order.
- A logistics mission-plan manifest links the parent survey, identity registry,
  validated task, planner configuration, route bundle, semantic station order,
  and corresponding candidate order.
- An execution-evidence manifest contract links one unique attempt to its
  mission plan, controller profile, route certificate, event log, result
  summary, timestamps, and outcome. The contract and immutable writer exist,
  but the segment runner does **not yet emit this manifest automatically**.

All hashes prove byte/configuration identity, not sensor accuracy or physical
safety.

## Exact Task Order

Survey order and logistics order are separate decisions. A survey may use
Held-Karp to reduce inspection travel. A logistics route must use
`--route-purpose logistics` and repeat `--fixed-station-order` in the exact
order stored in the validated task snapshot. Planning fails if the CLI order,
task snapshot, identity registry, catalog candidate set, map bundle, or parent
survey disagree. Replanning and retry logic preserve the same order; they do
not invoke TSP reordering.

`run_logistics_mission.py --dry-run` can fetch/validate server state and publish
an immutable task snapshot. `MissionController` can validate strict QR events
and produce navigation-neutral dispatches in that order. Neither component
invokes the ROS follower.

## Route and Command Safety Contracts

`plan_arrival_catalog_route.py` writes an execution route certificate alongside
the route. It binds the route bytes, frame, route kind, waypoint count, tracking
tube, exact-vertex pursuit policy, command-owner identity, map bundle, and
candidate snapshot. Task-ordered static-route dry runs additionally require the
mission-plan manifest, route-bundle descriptor, runtime map-bundle descriptor,
matching robot ID, exact selected leg target, and a still-fresh task-plan check.
The mission manifest is therefore the executable publication commit root; an
orphaned route/certificate/diagnostics trio fails admission. Survey static
routes have a separately named simulation-only escape hatch.

At runtime the follower checks the live base pose and commanded pursuit chord
against the active certified segment. Leaving the configured route tube, or
attempting an uncertified lookahead shortcut, produces a fail-closed stop. This
does not compensate for an incorrect map, localization estimate, footprint, or
tube radius. Route-tube departure is terminal for the current physical
authorization and never enters transient-obstacle recovery. Continuing requires
a separately resealed route, a new no-motion dry-run/preflight, and a new typed
`RUN`; the stopped process does not auto-relaunch or reuse the previous motion
authorization.

The same tube is included during planning around the active target stand and
around every frozen non-target stand. A catalog arrival is rejected when its
standoff lies outside the bare body envelope but inside the body-plus-tube or
LiDAR-plus-tube envelope.

`cmd_vel_guard.py` defines a separate lease/epoch/sequence/freshness state
machine that returns zero velocity for missing, expired, replayed, stale,
future-dated, or wrong-owner commands. It is currently a tested pure contract,
**not a running ROS guard or `twist_mux` replacement**. Existing graph-level
publisher checks cannot provide the same crash-independent guarantee.

## QR, Cargo, and Fleet Contracts

- Strict QR events bind robot and station identity, source/frame, observed and
  receipt clocks, image quadrilateral hash, multi-sample consensus hash,
  confidence, and calibration hash. Validation rejects stale, future, replayed,
  low-confidence, low-consensus, miscalibrated, or identity-inconsistent events.
- The current onboard camera node remains passive and writes the older scan
  evidence. It does not yet produce strict QR events or drive mission state.
- Carrier profiles validate unloaded/loaded footprint radii, robot mass,
  payload mass limit, and required retention. Puck custody uses monotonic fencing
  tokens for claim, load, delivery, loss, and release; another robot or stale
  token cannot mutate the claim. A reported loss blocks further transport
  motion until operator recovery.
- Station and conflict-zone permits expire, renew only with the current fencing
  token, and reject clock rollback. Central right-before-left arbitration uses
  approach geometry and deterministic request-time/robot-ID ties. Collision
  checks fail closed on missing/stale peer state and evaluate swept effective
  footprints, including the loaded radius.

The cargo and fleet items above are in-memory, ROS-free models. There is no
carrier sensor/actuator adapter, durable custody service, fleet communication
transport, or active two-robot coordinator node yet.

## Migration Gate

Do not enable real mission motion until every item in the current stage passes.

1. **Sealed simulation:** Run the full detector-produced candidate snapshot →
   survey manifest → server task snapshot → fixed-order mission plan → certified
   route chain without world-truth candidate coordinates. Preserve all attempts,
   including retries and safety stops, and generate execution-evidence manifests
   automatically.
2. **Passive real sensing:** The namespaced observe-only adapter now uses
   `CameraInfo`, a sealed `base <- camera_optical` transform, scan-stamped TF
   into the stable map frame, LiDAR target checks, image rectification, and a
   recorded calibration hash. Clear this stage only after its candidate, axis,
   QR, map, and AMCL evidence has been compared with physical measurements.
3. **Independent motion authority:** Deploy and test a real ROS command mux/guard
   implementing the lease contract. Process crash, lease loss, clock rollback,
   stale/future command, competing publisher, TF loss, and scan loss must all
   result in zero velocity independently of follower cleanup.
4. **Unloaded single robot:** Use `run_unloaded_segment.py` so the same sealed
   profile supplies namespace, frames, footprint, speed limits, and topics in
   dry run and motion. Validate the frozen map, route certificate, localization
   uncertainty, and route-tube margins before each leg.
5. **Loaded single robot:** Measure the physical carrier and loaded footprint,
   payload mass, retention, stopping distance, and acceleration limits. Wire
   positive load/unload/loss acknowledgement and custody fencing into the mission
   runtime before carrying a puck.
6. **Two robots:** Add authenticated namespaced status exchange and the active
   coordinator. Validate permit expiry/fencing, occupied-station waiting,
   right-before-left, certified holding poses, communication loss, coordinator
   restart, deadlock recovery, swept loaded footprints, and non-theft before a
   shared physical run.

At every physical stage: clear the arena, keep an operator beside each robot,
keep Ctrl+C and physical stop access ready, keep a separate exact-topic zero
`Twist` terminal ready, verify one localization owner and exclusive velocity
ownership, run `run_single_station_segment.py --dry-run`, and capture a distinct
`scripts/common/run_with_bundle.sh` evidence bundle. The bundle records evidence;
it is not a safety gate.

## Remaining Blockers

- Physical validation of the real synchronized viewpoint/axis adapter,
  measured calibration/extrinsics, and repeated passive-survey evidence.
- Strict QR-event production from the onboard camera and mission-controller to
  follower wiring.
- Active runtime attestation of the selected Gazebo world or physical site and
  live `/map` contents, rather than only checking the intended frozen
  descriptor and frame.
- Durable sequential dispatch/progress state with post-arrival QR confirmation;
  the generic multi-leg simulation wrapper is not a logistics executor.
- Automatic execution-evidence manifest creation covering every attempt.
- Deployed ROS command guard/mux and watchdog.
- Measured carrier, retention/load sensing, loaded dynamics, and custody runtime.
- Persistent/networked fleet coordinator and namespaced two-robot integration.
- Repeated blind simulation and passive-real evidence with predefined acceptance
  thresholds; one selected successful Gazebo sequence is insufficient.

Until these blockers are closed, use the new contracts for offline tests,
sealed simulation, and passive real-data collection only.
