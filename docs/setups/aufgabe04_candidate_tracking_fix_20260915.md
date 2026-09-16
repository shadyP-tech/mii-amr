# Candidate head tracking correction — 2026-09-15

The named-candidate observer now retains a verified complete head as the next
image's search region, using the same current-pixel 3D fitting path that worked
in the viewer. QR appearance does not clear this search context.

## Implementation

- `observer/candidate_head_tracking.py` owns the bounded head crop and camera
  pose hint. Seeds must pass current geometry, model, candidate association and
  observer freshness. Its separate two-second search lifetime permits a fresh
  400 ms acquisition result and 600 ms image intervals without relaxing the
  freshness of any admitted angle. Candidate, calibration, model, motion epoch,
  movement, expiry, invalid geometry and failed association invalidate it.
- `perception/stand_axis/physical_head_pipeline.py` now uses this hint for named
  candidates. Four borders and pose are fitted from each new image. A failed or
  ambiguous tracked fit cannot retry another border choice on that same image.
- `observer/tracked_head_registration.py` supplies a fresh current-fit crop
  proof, independently checked against the original candidate projection and
  current unique scan association. It does not mislabel old corners as a new
  proposal. Both faces use this proof; backside still requires complete current
  pixels and explicit current marker absence. Off-center backside classification
  uses the newly fitted head center, with original candidate bounds enforced
  separately. Neck validation and QR size are not angle gates.
- `observer/scan_witness_collection.py` collects scans at callbacks and a bounded
  TF retry timer, independently of camera fitting. Each scan uses its own exact
  scan-time transforms and robot pose. `scan_witness_buffer.py` and
  `scan_target_persistence.py` retain bounded independent witnesses and recheck
  them against the current head ray. Three distinct contiguous real witnesses,
  actual bridging returns, freshness, motion and competing-cluster rejection
  remain required. Expired pending TF clears previous witness history.

The node retains the existing seven-sample consensus, current QR-to-head binding,
publication freshness and route/motion admission checks. No new CLI flags are
required. Generic viewer-tracker predictions cannot bypass candidate tracking.

## Validation

**417 tests and 670 subtests passed across 40 focused suites**, after removing
an accidentally duplicated imported test fixture. Python 3.10 syntax and
`git diff --check` passed. Tests cover slow fresh acquisition, seven-frame front
admission, shifted crops/intrinsics, front-marker vetoes, off-center backside
classification, stale/clipped/ambiguous inputs, motion/context changes, independent
scan arrival, exact-TF retry ordering and preservation of camera TF diagnostics.

The final source snapshot was replayed in place on mii001 using the actual
`PassiveRealViewpointNode._process_latest`, original first-inspection compressed
images, recorded CameraInfo, exact saved TF and raw scans from
`stand_explore_exact2_camera_all5_20260915T143600Z`.

The first accepted recommendation is **QR_003 at frame_000017**, source
14:41:19.544 UTC. Its seven accepted samples begin at frame_000010 and span
**3.184 seconds**. Every sample passes newly computed geometry, current unique
LiDAR association, QR-to-head binding, freshness and temporal consistency.
The initial acquisition takes about 352 ms; the following tracked frames take
about 24–25 ms in this replay. The original run never reached seven simultaneous
accepted angles at this inspection point.

[Derived replay evidence and exact source hashes](aufgabe04_candidate_tracking_replay_20260915.json)
record the seven samples. All ten replayed source hashes match the final code.

## Scope and remaining hardware evidence

The patch was supplied through an in-memory import overlay on a read-only mount.
Original images and recordings remained on mii001. No deployment, ROS command or
robot motion occurred. This is saved-data admission evidence, not a live robot
success: the replay clock uses recorded processing-start ROS time plus measured
replay execution and excludes live DDS and diagnostic file-write delays.

Only saved selected scans were available; unrecorded independent callbacks were
not fabricated. The scan collector is covered by offline tests and still needs
live timing evidence. QR-ray binding continues to reject raw ambiguous clusters;
a witnessed fragmented scan can support the head angle but cannot independently
bind QR identity on that frame. The successful seven-sample window uses accepted
current QR bindings throughout.

The earlier fallback route-clearance rejection is unchanged. Hardware validation
should confirm first-view recommendation completion and normal parent handoff
before making claims about later candidates or the full logistics mission.
