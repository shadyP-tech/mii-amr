# Original opposite-side camera failure, 1 October 2026

Run: `stand_explore_exact2_camera_all5_20261001T141116Z`, candidate
`survey_candidate_0001`, `camera_lidar_attempt_02`, deployed revision
`144f95425e36256b2db82b84cc555e6864c726f1`.

`frame_000005.jpg`, `frame_000016.jpg`, and `frame_000031.jpg` are byte-identical
copies of the run's original JPEG-compressed camera messages. Their SHA-256
values remain in the original frame records in `input.json`.

The compact JSON preserves complete parsed source artifacts and frame records:

- The actual seven-sample backside observation, its source/arrival candidate
  frame projections, and all three candidate snapshots.
- Frames 3–5, 14–16, and 29–31, including original CameraInfo, LaserScan, TF,
  timestamps, diagnostics, and outcome records.
- The identical sealed camera profile already authenticated for the earlier
  October 1 endpoint fixture. Its digest equals this run's recorded calibration
  digest. The stand model is referenced by repository path and SHA-256.
- Provenance records with source paths, byte hashes, and canonical payload hashes.

The original backside source has **no validated metric center**. The fixture
helper rebases only path-bearing projections and recomputes their hashes; it
retains the real orientation-only observation. Three original current scans
are reconciled by production code, including the authenticated position-epoch
recovery. It does not invent an accepted head, QR observation, or robot pose.

For frames with a recorded search envelope, replay time is original scan receipt
plus its recorded age. Frame 29 expired later in image processing; its original
selection time is recovered from its recorded monotonic selection timestamp and
same-frame ROS/monotonic outcome pair. No sensor timestamp or range is shifted.

These images caused the native QR-outline gate to reject the live crop. The
workstation's WeChat decoder independently read `Start` without usable corners
from all three frames during the audit. The portable integration test mocks only
that payload result because the desktop OpenCV installation lacks WeChat; head
border detection, scan registration, foreground/background exclusion, receipt
construction/loading, and retained-angle processing use production code.

The structural test freezes the head detector's monotonic budget clock to avoid
host scheduling changing its result. It replays the source clock explicitly and
tests both post-decode and pre-publication expiry. This test is not end-to-end
runtime latency evidence; the workstation replay must measure that separately.
