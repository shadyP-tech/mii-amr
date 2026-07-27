# Aufgabe 04 Two-Robot Runbook

**Two-robot runtime is not implemented and is not cleared for physical use.**
`run_two_robot_mission.py` intentionally exits. The current fleet code is a
ROS-free policy foundation for offline and simulation testing only. Complete
the single-robot stages in
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md) before adding a
fleet transport or enabling shared motion.

## Implemented Pure Contracts

- Station leases have expiry, renewal, monotonic-clock checks, and fencing
  tokens. An expired or stale holder cannot release/renew a later lease.
- Shared conflict-zone reservations are exclusive, expiring, and fenced.
- Central right-before-left arbitration derives the right side from approach
  poses and yaw. Request time and robot ID provide a deterministic tie break.
- Missing, invalid, stale, or future peer state fails closed. Collision checks
  evaluate current and constant-velocity swept circular footprints; a loaded
  robot must provide its larger loaded radius.
- Puck custody uses a fenced owner and explicit available/claimed/loaded/
  delivered/lost states. Another robot and stale claim tokens cannot mutate the
  puck. A loss blocks transport pending operator recovery.
- Carrier profiles distinguish unloaded and loaded footprint, robot/payload
  mass, payload limit, and required retention.

These contracts are in-memory. They are not durable consensus, authentication,
network partition handling, ROS messages, trajectory reservation, or physical
stop enforcement.

## Required Runtime Design Before Simulation Motion

1. Give each robot unique node names, namespaces, camera/scan/odom/cmd topics,
   TF frame prefixes, command-owner identity, run ID, and evidence directory.
2. Add authenticated, timestamped robot-status exchange with explicit clock
   identity, velocity, load state, base footprint, loaded footprint, and current
   permit/custody fencing tokens.
3. Run one active coordinator authority with persistent fence counters. A
   restart must not reuse an older fence. Communication loss, stale state, or
   clock rollback must revoke motion authority and command zero velocity.
4. Define conflict-zone polygons and collision-certified holding poses outside
   every station/zone. A `WAIT`/`YIELD` decision is safe only if the robot can
   stop at its holding pose within the measured loaded stopping distance.
5. Require the station lease before entering an approach zone and a
   conflict-zone permit before crossing. Renew only with the current fence and
   release only after the complete footprint has left the resource.
6. Keep the immutable server task order per robot. Fleet waiting/replanning may
   delay a step but must not reorder it.
7. Bind puck identity and custody to the server task. Loading another robot's
   claimed puck, moving with unknown retention, or continuing after loss must
   fail closed.

## Mandatory Offline and Simulation Cases

- exactly one right-before-left winner for perpendicular approaches;
- deterministic resolution of ambiguous/cyclic arrivals without livelock;
- occupied-station waiting and safe resumption after fenced lease expiry;
- stale renewal/release, coordinator restart, clock rollback, replay, delayed
  status, and network partition;
- missing pose/yaw/velocity/footprint and future or stale timestamps;
- swept collision prediction with unloaded and loaded footprints;
- both robots requesting one station/zone and one puck concurrently;
- dropped/lost puck, stale custody token, and attempted puck theft;
- command-owner loss, follower crash, and competing velocity publisher;
- no holding pose or insufficient loaded stopping distance.

Do not proceed on “best effort” when any required state is unknown. The safe
result is both robots stopped outside the resource until fresh state and a
valid fenced permit exist.

## Physical Gate

Before any two-robot physical run, each robot needs a successful sealed,
server-ordered, QR-confirmed, bundle-wrapped single-robot mission using the same
hardware profile. The deployed coordinator, independent command guard/mux, and
all mandatory cases above must then pass in two-robot simulation.

For the physical run, keep an operator and physical stop beside each robot,
separate exact-topic zero-`Twist` terminals, no competing Nav2/custom goals,
fresh localization/scan/TF, and distinct
`results/real_runs/<run_id>/` bundles. Bundles are evidence capture only; they
do not replace permits, preflight, watchdogs, or physical stops.
