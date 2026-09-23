# Opposite-side QR admission with retained orientation

After a certified backside observation triggers the opposite-side route, the
parent now reprojects that original orientation into the admitted arrival frame.
This includes its uncertainty interval and source hashes. Corrective arrival
alignment uses the same handoff. The observer receives the retained evidence and
the arrival candidate snapshot through internal arguments; the autonomous run
command needs no new flags.

The opposite-side observer decodes identity before physical head fitting. One
fresh, unambiguous QR payload from the validated candidate crop completes
discovery immediately, including when the decoder returns no usable corners.
There is no frontside angle fit, additional axis consensus, or geometry grace on
this branch. The robot must be on the opposite side for the retained orientation
interval; exact sensor-time transforms, stationary checks, calibration matching,
source freshness, and conflicting-identity rejection remain in effect.

The target search starts from a unique current LiDAR return. A lightweight QR
outline detector associates a complete symbol with that return using current
pixel scale, bearing, range, exact transforms, and the existing registration
limits. It does not fit a stand angle. Overlap with a mapped neighbor is resolved
only when conservative depth intervals put that neighbor behind the supported
foreground QR. The decoder then samples the complete foreground quadrilateral
in isolation; it never receives a rectangle cut through the target QR. An
unresolved overlap produces `opposite_identity_crop_conflict` for bounded
recovery. Without overlap, a validated rectangular crop still supports payload
results with no usable decoder corners.

The route continues to use the retained orientation and the LiDAR candidate
center validated through the source registry and fresh planning-frame projection.
Arrival centering is enabled on the retained-angle branch. If identity is still
unresolved, the current QR outline and associated range can supply a calibrated
centering advisory without another stand-angle estimate. Existing limits remain:
two turns, at most six degrees per step and twelve degrees total, exact image-time
odometry, stopped post-turn evidence, and a strictly newer sensor tuple. The
existing destination-framing check can decline an unproductive turn. A successful
QR decode takes precedence and completes immediately without an unnecessary turn.

Retained orientation is reprojected again after centering or bounded view recovery;
its original sample evidence and uncertainty are preserved. No camera measurement
silently rewrites the frozen LiDAR candidate. New identity acquisition states enter
the existing typed timeout recovery, preserving conflict and process-cleanup checks.
Timeout diagnostics distinguish missing identity from missing angle consensus.

Implementation responsibilities:

- `artifacts/retained_backside_orientation.py`: validate retained orientation,
  source/frame lineage, and opposite-side eligibility.
- `real_robot/observer/opposite_identity_crop.py`: current target crop and
  complete-symbol isolation, depth separation, and cornerless text association.
- `real_robot/observer/opposite_target_support.py`: current QR outline, pixel
  scale and unique scan association, shared with centering receipt validation.
- `real_robot/candidate/retained_orientation.py`: reproject the original angle
  into each admitted post-turn/recovery frame.
- `real_robot/observer/opposite_identity.py`: bounded decoding and submission to
  the existing fresh-observation commit path.
- `artifacts/qr_verified_observation_pose.py`: version 2 discovery receipts with
  retained orientation; version 1 behavior remains supported.
- `real_robot/candidate/inspection_adapters.py`: arrival handoff and observer
  request wiring. The discovery catalog preserves the angle and its provenance.

The receipt remains discovery-only (`facing_ready=false` and
`motion_authorized=false`). It records a historical orientation and the actual
robot observation pose; future motion still needs its normal admission.

Validation includes a captured image/scan/TF regression fixture from run
`20260922T150953Z`, retained-angle centering across a localization-frame change,
full-symbol/depth checks, source freshness and conflicting identities, timeout
classification at the parent boundary, and catalog propagation. The local OpenCV
build lacks WeChat, so the deployed-backend decode test is skipped locally and is
covered by a separate read-only replay in the workstation's OpenCV 4.5.4 container.

The regression suite passed 223 tests and 297 subtests, with one deployed-backend
test skipped locally. The corrected production crop path decoded `Start` in 11
of 12 sampled saved frames (median crop/detection/decode time 35.8 ms); one lacked
accepted current target support and remained unresolved. The old crop path
decoded none of those 12. See
`results/implementation_checks/opposite_identity_audit_20260922T150953Z/corrected_deployed_replay.json`.
This is recorded-data validation; a new live robot run is still required.
