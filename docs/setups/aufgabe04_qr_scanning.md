# Aufgabe 04 QR Scanning

QR support currently consists of a passive ROS scanner, immutable task-server
snapshots, and stricter ROS-free evidence/mission contracts. These pieces are
not yet one live mission pipeline. See
[`aufgabe04_sim_to_real_gate.md`](aufgabe04_sim_to_real_gate.md) before using QR
data to authorize motion.

## Passive Onboard Scanner

`scripts/aufgabe04/qr_scanning/onboard_camera_node.py` subscribes to the
configured compressed-image topic, decodes identifiers, and appends scan rows
to `results/aufgabe04/qr_scans.csv`. It never starts a mission or publishes
velocity.

Before collecting evidence, verify the resolved namespaced camera topic,
lighting, focus, QR size/range, image timestamp, and camera calibration on the
actual TurtleBot. A decoded text row by itself is not arrival confirmation or
motion authority.

## Strict QR Evidence Contract

`scripts/aufgabe04/qr_scanning/events.py` defines the required future mission
event. One `QRObservationEvent` binds:

- unique event and robot IDs;
- QR ID, canonical station ID, and candidate UID;
- observation and receipt timestamps plus clock ID;
- source topic/adapter and camera frame;
- confidence and calibration SHA-256;
- a hashed, ordered, convex four-corner image quadrilateral;
- a hashed multi-sample consensus window and agreeing sample IDs.

Validation rejects wrong-robot, stale, future, replayed, delayed,
low-confidence, low-consensus, miscalibrated, wrong-clock, or identity-mismatched
events. The identity registry must map candidate, QR, and server station IDs
one-to-one; aliases from different stations may not overlap.

The current onboard scanner does **not** emit this strict event. A real camera
adapter still needs to capture geometry/consensus/calibration evidence and
translate the file-backed station identity registry into the mission event
registry without changing its mapping.

Create that reviewed one-to-one registry from the frozen detector snapshot
before collecting QR mission evidence. Repeat `--mapping` for every candidate:

```bash
python3 scripts/aufgabe04/stations/create_station_identity_registry.py \
  --candidate-snapshot results/aufgabe04/detected_stations/candidate_snapshot_HASH.json \
  --mapping detected_stand_00=A=station_A \
  --mapping detected_stand_01=B=station_B
```

The ROS-free command validates complete candidate/QR/server identity and emits
an immutable `station_identity_registry_<full-content-hash>.json` beside the
snapshot by default. Review the printed mapping artifact before referencing it
from task snapshots or future strict camera events.

## Validated Server Task Snapshot

The existing task CLI is dry-run-only. It accepts a supplied QR ID, validates
fresh FastAPI status/plan data, preserves the remaining server order, and
writes an immutable content-hashed task snapshot:

```bash
python3 scripts/aufgabe04/run_logistics_mission.py \
  --dry-run \
  --robot-id robot_1 \
  --qr-id A \
  --station-identity-registry results/aufgabe04/detected_stations/station_identity_registry_HASH.json \
  --validated-task-json results/aufgabe04/task_snapshots/task_MISSION_ID_HASH.json \
  --print-task
```

The snapshot binds robot/mission identity, current and target stations, cargo,
source plan hash, source timestamps, validation timestamp, plan-step index, and
the exact `ordered_station_ids` plus its order hash. Use a content-derived
filename; a different task cannot overwrite an existing snapshot.

This CLI argument is not proof that the onboard camera observed the code. It
does not construct `QRObservationEvent`, invoke `MissionController`, or invoke
the follower. For migration evidence the strict camera event, server snapshot,
identity registry, dispatch, and post-arrival confirmation must be linked in
one attempt log.

## Mission-State Contract

The ROS-free `MissionController` accepts a fresh strict initial QR event and a
fresh validated server task, copies the server order once, and emits
navigation-neutral dispatches. Retry dispatches retain the same station index
and order. Advancement requires a new post-dispatch QR event for the expected
station; wrong, pre-dispatch, or replayed observations do not advance state.

The controller has bounded navigation attempts and confirmation rejections, but
has no ROS, camera, HTTP, filesystem, or follower integration. Keep it in
offline/simulation tests until that adapter exists and the migration gate is
cleared.
