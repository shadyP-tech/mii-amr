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

The crop comes from a unique current LiDAR return in the existing candidate
search envelope. It is narrowed around the projected head and clipped away from
other mapped candidate head volumes, including their positional uncertainty.
An ambiguous or unusable crop cannot bind text. This association is scoped to
the mapped candidates and calibration; it does not repair incorrect stand
locations or prove that an unmapped neighboring object is absent.

Implementation responsibilities:

- `artifacts/retained_backside_orientation.py`: validate retained orientation,
  source/frame lineage, and opposite-side eligibility.
- `real_robot/observer/opposite_identity_crop.py`: current target crop and
  cornerless text association.
- `real_robot/observer/opposite_identity.py`: bounded decoding and submission to
  the existing fresh-observation commit path.
- `artifacts/qr_verified_observation_pose.py`: version 2 discovery receipts with
  retained orientation; version 1 behavior remains supported.
- `real_robot/candidate/inspection_adapters.py`: arrival handoff and observer
  request wiring. The discovery catalog preserves the angle and its provenance.

The receipt remains discovery-only (`facing_ready=false` and
`motion_authorized=false`). It records a historical orientation and the actual
robot observation pose; future motion still needs its normal admission.

Validation: 157 tests and 111 subtests passed across opposite identity,
candidate approach/capture/inspection, QR discovery and fallback, observer
processing/centering, and bounded orientation/frame projection. Tests exercise
first-frame cornerless completion without head fitting, late/conflicting
decodes, neighboring crop overlap, changed evidence, frame rotation,
uncertainty retention, observer arguments, and catalog propagation.

This has not been validated on the real robot. It removes the corner and fresh
angle requirements on the certified opposite-side branch; it does not establish
that run `20260922T141529Z` would now complete. That run also showed a discrepancy
between the stored candidate center and its observed camera position, which can
still affect crop validity and decoding.
