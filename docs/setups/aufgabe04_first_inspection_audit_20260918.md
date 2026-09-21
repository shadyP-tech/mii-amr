# First candidate inspection audit — 18 September 2026

## Finding

The two inspected missions never started their first camera observer. Their
initial stopped-arrival receipts rejected the robot-to-candidate bearing before
head-frame angle estimation or QR decoding could run. The camera debug recording
at the same inspection point contains usable head borders and a readable QR.
These runs therefore do not establish a regression in the visual detector.

Both mission bundles identify clean commit
`d229b0475805332007534f17767ead6203ddeab7`. No robot motion or production code
changes were made during the initial read-only audit. The subsequent repair is
described below.

## Mission evidence

Artifacts are under
`results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_<timestamp>/`,
with terminal logs and motion traces under `results/real_runs/`.
The first candidate is `000_survey_candidate_0003` in both runs.

| UTC run timestamp | Arrival range | Absolute bearing error | Admission |
| --- | ---: | ---: | --- |
| 20260918T151004Z | 0.4274 m | 3.2436° | Rejected: limit 3° |
| 20260918T151752Z | 0.5471 m | 3.7896° | Rejected: limit 3° |

Both ranges passed. The source is each candidate's
`candidate_arrival_admission.json`. Neither mission contains a
`camera_lidar_attempt_*` directory, observer status, or inspection progress receipt.

The earlier run attempted alignment, but its initial alignment child and first
startup reseal both stopped with `TF transform unavailable: map <- odom`.
The second reseal passed dry preflight; the parent log ends after launching its
wrapped execution. It does not record a final mission outcome.

The later run's alignment child completed at 15:23:06.416 UTC. Its trace contains
actual angular commands, so the approximately 1 mm translational distance does
not mean the correction was a no-op. The child bundle completed at 15:23:07.
There is no subsequent `alignment_01/arrival` receipt, camera attempt, parent
traceback, or completed parent manifest. Remote log tails were rechecked and
contain no later explanation. An interruption, a blocked parent/preflight, or
another unrecorded failure cannot be distinguished from these artifacts.

## Why a completed approach can fail arrival

`controller_config_for_route_kind()` in
`scripts/aufgabe04/navigation/control/driving_behavior.py` already clamps
detected-stand terminal heading tolerance to 3°. This is not a loose 5° or 14°
controller tolerance.

However, the follower checks the planned endpoint yaw in its sealed execution
frame. `evaluate_candidate_arrival_admission()` checks the direction from the
actual stopped position to the freshly projected candidate. Position error and
localization changes affect the latter. In the later run, the arrival pose is
2.40° from the original planned yaw but 3.79° from the actual candidate bearing.
Equal numerical tolerances do not guarantee equal admission outcomes.

`inspection_adapters.py::admit_corrected()` handles this rejection before
`execute_candidate_inspection()` starts capture. Consequently the camera-based
centering adapter is also unreachable until arrival admission succeeds.

The latest commit's LiDAR inspection hints returned empty `candidate_views` in
both selection logs. The scan-boundary persistence change operates inside the
observer, which neither run reached. Neither change is implicated by these logs.

## Recording evidence

Recording:
`results/aufgabe04/stand_axis_debug_recordings/recording_20260918_172449_925160643`.
It contains 83 source images at 800 × 600, beginning at 15:24:49.885 UTC.

- 67 recorded model results accept the measured head geometry.
- 66 remain fresh, usable displayed angle estimates; one is obsolete.
- Median yaw is 28.4984°, ranging from 27.9977° to 29.1703°.
- 14 frames reject ambiguous nearest-LiDAR candidates and two reject stale or
  unsynchronized scan input. These are target-selection/timing failures.
- Color masking and adaptive foreground gating are enabled. The successful
  results explicitly report current measured borders and metric pixel-size
  screening. This demonstrates that the combined processing works for this
  red front-side view; it does not isolate the contribution of each filter.
- `no_qr_decode=true` on all 83 rows. The recorded QR-negative flags are therefore
  not evidence of decoding failure.

The project's `detect_qr_observations_bgr()` decoded `QR_003`, with actual QR
corners, from all 83 original images in a separate full-image replay. Median
decode time was 30.6 ms; maximum 396 ms. Replay used local OpenCV 5.0.0 with no
live processing deadline. This establishes image readability, not equivalence
to the robot's backend, live freshness, candidate association, or QR admission.
The viewer also uses `head_target=nearest`; it does not reproduce the mission's
frozen candidate binding and complete observer receipt requirements.

Replay script, per-frame results, and log:
`results/implementation_checks/first_inspection_audit_20260918/`.
Run from the repository root with `PYTHONPATH=.` and an OpenCV-enabled Python.

## Repair direction

Prioritize the arrival-to-observer handoff. Terminal alignment should account
for bearing from the stopped position, with sufficient margin for the fresh
arrival check. A bounded observation/centering admission path is another design
option, but must retain candidate association and certified turn requirements.
Simply loosening head-border or QR checks cannot fix a camera that never starts.

Add durable parent events around alignment return, fresh arrival preflight,
observer launch, and exception/interruption termination. The later run's exact
post-alignment failure remains unresolved without that evidence. Address the
earlier run's TF startup failures separately. Preserve the current visual
detector while validating this handoff and avoid claiming an end-to-end fix
from the offline replay alone.

## Implemented repair — 21 September 2026

Local inspection now explicitly requests passive centering acquisition when its
certified camera-centering effect is available. The acquisition bearing envelope
is 6° (the existing maximum single centering step). The receipt preserves the
ordinary 3° decision as `strict_arrival` and marks a bearing-only acquisition
admission with `acquisition_only=true`. It does not declare the camera centered
or authorize motion. Range checks, fresh stationary localization, candidate
projection and binding, live observer association, QR validation, and certified
turn checks remain in effect. Calls outside this local centering path retain
ordinary arrival admission. Larger bearing misses retain bounded route alignment.

Thus both recorded bearing misses can start camera inspection at the current
view without the additional alignment route that previously blocked them. Any
turn still requires fresh camera/scan evidence and the existing bounded child
permit; a usable QR or joint observation can complete without a turn.

Each candidate now records `inspection_handoff_events.jsonl`, with started,
returned, and failed boundaries for arrival admission, inspection motion, and
observer capture. Exceptions and Python interruptions are recorded and
re-raised. A process killed without cleanup still leaves its last started
boundary; this does not promise a final event after a hard kill.

Validation: 134 tests and 85 subtests passed across autonomous candidate approach,
arrival admission, inspection execution, camera centering (observer, child and
runtime), camera binding, and driving behavior. New regressions cover both
recorded bearing misses reaching first capture without route alignment; range,
capability and 6° envelope rejection; retained alignment for a 30° miss; and
durable TF-failure/interruption evidence. No real robot run or deployment was
performed. The earlier TF outage and the later unrecorded termination are not
claimed to be repaired as independent infrastructure failures.

## QR fallback follow-up

The workstation's two newest `stand_explore*` runs were rechecked: they remain
`20260918T151004Z` and `20260918T151752Z`. Both have zero camera-attempt directories,
zero QR observation-pose receipts, and zero inspection progress receipts. Their
requested inspection views explicitly say `purpose=arrival_alignment`,
`view_index=0`, and `stand_axis_authorized=false`. They did not select another
view after rejecting a successful live QR decode. They tried to correct arrival
before starting the observer. The 83 successful QR decodes previously reported
were a separate offline replay of the subsequent debug recording; they were
not evidence available to those running missions.

The existing fallback is enabled by the autonomous runner's
`--qr-observation-pose-json` argument. Its default geometry grace is zero seconds:
one current independently associated decode with QR corners can suffice even
without a head-angle result. It still requires admitted fresh image/scan and
stationary-epoch evidence, unique identity, valid localization/binding, a valid
receipt, and successful observer completion. Text decoding alone is not that
receipt.

`inspection_execution.py` treats `qr_observation_pose_path` as terminal discovery
success before considering further local views. The approach coordinator binds
and stores `candidate_qr_discovery.json` without facing validation. This accepts
the observation pose for discovery, but does not establish stand orientation or
authorize a later facing/puck approach. The fallback does not require another
inspection point merely because the head angle is unavailable.

Follow-up validation: 32 tests and 39 subtests passed in QR observation pose,
transport, catalog boundary, and local inspection suites, including the explicit
test that a QR-only receipt completes without another move. One broader producer
test failed before QR logic because its stale fixture supplies `cv2=object()` to
the rectification path. An in-memory retry with real OpenCV then exposed missing
`head_depth_m` in the same fixture. This is an unvalidated integration test, not
evidence of a live QR rejection in either recorded run. No additional production
changes were made for this follow-up.
