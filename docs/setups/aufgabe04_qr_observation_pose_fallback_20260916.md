# QR-confirmed observation pose fallback

Autonomous camera exploration now accepts a distinct discovery result when the
robot can read a uniquely associated candidate QR but cannot estimate the stand
head angle. It records the robot's original, stopped image-time pose, leaves
`stand_axis_rad` null, and continues to the next candidate. The LiDAR stand
center and full obstacle snapshot remain unchanged.

## Admission and timing

The current frame first receives its bounded geometry attempt. A valid head
recommendation takes precedence. The default QR fallback requires one current
decoded symbol with valid corners and an independently accepted, unique
camera/LiDAR association. It does not require a head proposal, neck validation,
head/QR size agreement or an angle sample. Duplicate codes, conflicting identity,
motion-epoch reset frames, stale/unsynchronized sensors, missing exact-time
transforms and publication-age failures cannot produce a fallback receipt.

The autonomous runner passes `--qr-observation-pose-json` automatically; existing
exploration commands need no new flags. The passive observer also offers
`--qr-pose-fallback-delay-sec` (default `0`, range `0–10`). A positive optional
delay gives geometry additional time at the same stop and requires a fresh
accepted QR observation after that interval; it never republishes an expired
decode as current evidence.

## Modular implementation

- `artifacts/qr_verified_observation_pose.py`: hashed discovery-only contract,
  original robot pose and source stamps, QR quadrilateral, unique scan-cluster
  evidence, calibration/profile context and localization provenance.
- `observer/qr_observation_pose.py`: admission/reset policy and fresh terminal
  publication, independent of head-angle success.
- `observer/qr_observation_binding.py`: parent-side candidate, frame, profile and
  stream validation. Observer process cleanup and successful terminal status
  remain required.
- `candidate/inspection_execution.py`: same-pose discovery completion without
  issuing another local-view or opposite-side movement.
- `candidate/qr_pose_discovery.py`: original observation pose and admitted
  candidate-frame projection in a separate durable discovery receipt/catalog.
- `candidate/qr_goal_progress.py`: distinct QR counts across both evidence types,
  separate facing readiness, and duplicate quarantine across both types.

## Artifacts and completion

An accepted fallback creates `qr_observation_pose.json` in its camera attempt and
`candidate_qr_discovery.json` in the candidate directory. A completed mission
writes `qr_observation_pose_catalog.json` for fallback records and keeps only
geometry-backed records in `stand_facing_catalog.json`. Confirmed identities can
satisfy the five-stand discovery goal with a mixture of these outcomes.

The summary distinguishes `goal_completed`, `facing_ready_stand_count`,
`qr_only_stand_count`, `facing_complete` and `camera_geometry_complete`. Existing
one-candidate pilot limits remain effective. Fallback poses do not change the
source survey geometry or certify a logistics approach; incomplete facing
catalogs are rejected by logistics promotion. Planning, localization, process
and evidence-integrity failures retain their normal terminal behavior.

## Validation

Offline regression scenarios cover all five candidates using the fallback,
mixed outcomes, duplicate identities in both evidence orderings, preservation
of the original robot pose, reprojected camera-arrival frames after motion,
single-view/pilot completion, forged completion claims and wrong-candidate
receipts. Observer tests exercise actual processing with failed head acquisition
and successful bound QR decoding, plus stale/moving/conflicting negative cases.

The combined 123-file suite passed **1,287 tests and 1,161 subtests**, with one
backend-dependent test skipped. One unrelated test failed:
`test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`.
Its mocked odometry certificate lacks `odom_execution_certificate_sha256`.
Re-running that test with the HEAD versions of every changed production module
reproduced the same failure; the production certificate check was left intact.
The final six mission-reporting tests also passed after adding an explicit
mixed-discovery summary case. Python parsing and `git diff --check` passed.
Verification logs and the baseline commit are saved under
[`qr_pose_fallback_20260916`](../../results/aufgabe04/debug_audits/qr_pose_fallback_20260916/regression_results.txt).

No deployment, remote recording transfer or robot experiment is part of this
change. Live confirmation remains required for sensor timing and field behavior.
