# Candidate position recovery after stopped arrival

Implements the arrival correction identified in the audit of
`stand_explore_exact2_camera_all5_20260928T130911Z`. The affected fourth visited
stand is `survey_candidate_0005` (`candidate_003` in navigation run names).
Its arrival projection moved roughly 24 cm from the original survey map point;
the current visible head was about 300 pixels left of that projection.

## Behavior

The parent passes its existing, content-bound candidate frame projection into
each camera inspection. Preapproach and arrival continue to use fresh planning
frame admissions; this change does not silently replace the survey landmark or
alter route clearance. At stopped arrival, normal association is tried first.
Only an empty normal 15-degree acquisition envelope can enter position recovery.

Recovery requires a verified projection/snapshot pair, a localization-induced
candidate displacement of 8–35 cm, a recent stopped planning pose, and three
independent fresh scan/image timestamps within 1.5 seconds. The unchanged
candidate surface-range interval and a bounded 35-degree search must contain
one compact cluster of at least three beams. It must agree with the original
map hypothesis, stay within 35 cm of the projected candidate, and be separated
from every other candidate at both map hypotheses. Raw fragments are not joined.

The recovered cluster supplies a search region, not head corners or identity.
Viewer-style nearest-head acquisition uses that region. The measured head or
complete decoded QR must separately agree with the cluster using finite range,
calibrated camera translation and a narrow three-degree ray bound. QR receipts
replay the proof. Complete associated QR admission retains priority over motion
or further head-angle collection. Opposite-side search can consume the same
proof while preserving its retained backside orientation.

A validated recovery may request one coarse yaw-only arrival turn up to 30
degrees, followed by up to two fresh six-degree fine turns. Total measured travel
is bounded by 42 degrees, including stopping motion. This is explicitly included
in the new mission RUN scope. Older scopes retain their original authority and
cannot authorize this extension. Every turn still needs its own sealed permit,
newer live scan/odometry, stationary anchor, clearance checks, exclusive velocity
ownership and stopped-pose evidence. A failed turn cannot reset its budget.

The recorded scan plus later same-framing viewer center require 22.70 degrees
left. Full centering would enter the protected LiDAR scan boundary. The planner
therefore requests 16.70 degrees, preserving that veto and requiring a fresh
observation before another action. Full optical centering is not guaranteed if
the scan geometry makes it unsafe to verify the target there.

Five seconds of continuously fresh, recovery-proven but camera-unbound frames
produce `candidate_position_epoch_inconsistent`. A clean child exit with that
state enters the existing bounded inspection recovery; crashes, poisoned
identity evidence, stale frames and forced exits cannot use that shortcut.

## Modules and verification

- `observer/candidate_position_epoch.py`: projection validation, bounded cluster
  recovery and scan-safe coarse turn selection.
- `observer/target_reconciliation.py`: three-frame stationary proof.
- `observer/position_epoch_opportunity.py`: bounded unbound-target opportunity.
- Existing head/QR association modules consume the same proof.
- Existing centering planner, parent, permit and child preserve the cumulative
  motion budget and proof binding. Live target preflight now uses the same
  finite-range camera-offset calculation as observation and centering.

Recorded and synthetic regressions cover recovered blue-head acquisition,
ordinary-gate rejection, competing candidates, changed scans/projection hashes,
stale samples, robot motion, QR receipt validation, calibrated live preflight,
coarse-to-fine child execution, authorization scope and turn-budget preservation.
See the fixture README for the exact boundaries of recorded versus synthetic
replay. The saved late image and viewer corners are not simultaneous with the
three selected scans. An SSH host-key mismatch prevented fetching the exact
matching image. No ROS/OpenCV 4.5.4 replay, deployment or physical run was made.

Validation on the local offline Python/OpenCV environment: the scoped suite
passed **265 tests and 298 subtests**, with one skipped test. After the final
frame-binding and CLI-transport checks, the affected subset passed **27 tests
and 10 subtests**. `git diff --check` passed. Existing real-run CLI flags remain
valid; a new run creates the updated RUN scope and fresh recovery artifacts.
