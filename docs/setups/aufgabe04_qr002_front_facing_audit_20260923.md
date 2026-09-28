# QR_002 facing admission and retained backside orientation audit

Evidence inspected: run `stand_explore_exact2_camera_all5_20260923T144640Z`
and the latest viewer recording `recording_20260923_170442_538403382`.
The recording is later than the run, not a synchronized recording of it.
Artifacts were read from `mii001`; no robot nodes or motion were started.
This is an investigation, not an implementation or a hardware validation.

Local evidence root:
`results/implementation_checks/run_audit_20260923T144640Z/`.
The nested `results/aufgabe04/` tree preserves the source artifact paths.

## Findings

QR_002 was admitted as an identity at its initial camera inspection. It did
not receive facing geometry. It never triggered an opposite-side branch and
has no retained backside orientation in its QR receipt. Start is the candidate
that had a successful backside observation followed by an opposite-side QR
decode but failed retained-facing promotion.

### QR_002: acquisition works, association and angle admission do not

Source: `candidates/002_survey_candidate_0002/camera_lidar_attempt_00/`.
All 112 processed frames decoded QR_002 and contained accepted bounded head
orientation evidence. The acquired material rim is purple. The raw final
frame, rectified with its recorded camera calibration, shows the entire head
and QR. The head center is approximately (386,287), versus principal point
(405.87,300.71): there is a modest image offset, not a clipped target.

Strict angle results:

| Result | Frames |
| --- | ---: |
| Planar axis ambiguous | 65 |
| Yaw uncertainty too high | 46 |
| Strict measured-head angle accepted | 1 |

The one strict fit, frame 54, failed current head/LiDAR association
(`ambiguous_registered_camera_clusters`). It could not admit facing geometry.
The front-facing square supports multiple near-equivalent planar poses;
detecting its border is not equivalent to measuring a precise yaw.

The independent QR binding rejected 109/112 frames as
`finite_qr_range_not_unique`. Its broad registration envelope counted two
clusters even when the narrow head-association cone succeeded (70/112 head
associations succeeded). The recorded broad-envelope breakdown is:

| Two-cluster scan topology | Frames |
| --- | ---: |
| Inconsistent endpoint metadata | 96 |
| Seam not one sampling step | 8 |
| Valid full rotation, internal missing beam | 5 |

Example frame 1: the near-target returns are indices [0,1,2] and [215,216],
all approximately 0.525–0.529 m. Circular adjacency is disabled by the endpoint
metadata check. Example frame 32: [216,0,1,2] plus index 214, with missing beam
215, all approximately 0.525–0.529 m. These are consistent with fragmentation
of the same stand, but that inference cannot by itself replace identity proof.

`observer/qr_target_binding.py` requires a unique broad-envelope range before
applying the calibrated finite-range camera ray. `observer/qr_candidate_search.py`
explicitly does not use witnessed fragmentation. In contrast, the head path
can call the stopped scan-persistence resolver. The paths therefore have
different cluster admissibility rules. Fixing only head acquisition cannot
remove this QR bottleneck.

Only frames 35 and 41 had both head-bound QR and bounded geometry admitted;
the bounded orientation window reached two of seven samples. Both fell within
the 1.5 s geometry opportunity measured from the first accepted QR sample
(their image stamps differ by 1.233 s).

Frame 112 finally had independently associated QR identity, but the returned
QR polygon included bottom-right corner (446.46,358), below the actual head
bottom near y=341. The final rectified image shows that this is outside the
physical label. Consequently `current_head_qr_binding.py` rejected it as
`qr_outside_current_head`. This explains why identity completed without the
current-head bounded sidecar. Enlarging the head polygon or trusting the bad
corner would not be a valid geometry correction.

All 112 processed tuples were synchronized and TF-ready. Publication image
and scan ages were 0.132 s and 0.143 s. There were separate scan-witness TF
delays/expirations, so this does not establish that TF delivery was perfect;
the recorded admission blockers for QR_002 were the above association and
geometry gates, not an image/scan TF timeout at publication.

### Latest viewer recording confirms the geometry limitation

The 54 source images span 5.166 s. The purple head is fully visible; its width
is approximately 103 px. Recorded result reasons:

| Result | Frames |
| --- | ---: |
| Yaw uncertainty too high | 38 |
| Planar axis ambiguous | 1 |
| Scan stale or unsynchronized | 14 |
| Obsolete detector result | 1 |

There are no accepted precise-angle results. The 39 geometry-bearing results
retain bounded orientation evidence. For example, frame 0 has yaw standard
deviation 4.34 degrees against a 3-degree strict limit, while the conservative
orientation interval has half-width 13.94 degrees.

Viewer settings include `head_target=nearest` and `no_qr_decode=true`.
The recording confirms good border acquisition and weak precise-yaw
observability; it does not demonstrate successful QR-to-candidate admission.

### Start: angle ready before center proof

Source: `candidates/001_survey_candidate_0001/`.
The backside receipt has seven angle samples. On its final captured frame
(11), independent target reconciliation is still `collecting_stopped_target`,
sample count 1 of the required 3. The observer commits the backside angle
without `target_reconciliation` or `head_position_evidence`.

The arrival-frame retained record correctly contains:

- stand axis 1.391005645 rad;
- interval half-width 0.132788041 rad (7.608 degrees), seven samples;
- opposite outward face normal 2.961801972 rad (169.699 degrees);
- immutable source/projection bindings and `current_angle_refit=false`.

It has no `validated_target_center`. After decoding Start on the opposite
side, `retained_facing_status.json` records:
`retained facing requires validated center and bounded orientation`.
The interval exists: the missing center is the actual failed prerequisite.

This is a readiness-contract mismatch. `BacksideAxisObservation` accepts an
angle without a reconciled center, while `artifacts/retained_facing.py` requires
both for durable facing geometry. The angle was neither lost nor incorrectly
reversed. Merely adding pi to the stored axis will not repair this failure.

## Required correction

1. Give QR range registration and center reconciliation access to a shared,
   validated current-cluster proof, including bounded endpoint/internal-gap
   recovery backed by stopped scan witnesses. Preserve source indices,
   freshness, finite-range camera translation, candidate range limits and
   competing-candidate rejection. Do not globally enable circular adjacency
   for inconsistent scans or simply ignore the second cluster.
2. Retain a successful backside angle and its complete uncertainty while the
   independent center proof finishes within a bounded opportunity. Carry both
   through the authenticated frame projection. If a legacy retained receipt
   lacks a center, resolve the center independently at arrival; do not require
   another front-side angle fit or silently label the survey center measured.
   Bounded failure preserves QR discovery without fabricating facing geometry.
3. Once the complete current QR is uniquely associated with that candidate,
   select the opposite directed face from the retained orientation immediately.
   No new front-side head corners, precise-angle consensus or angle refit are
   needed on this retained path. Route and clearance checks still determine
   whether the resulting facing pose is usable.

For a directed backside outward normal beta:
`front_outward_normal = wrap(beta + pi)`.
The undirected stand axis is unchanged modulo pi; `-axis` is not the required
operation. In-place robot heading points toward the center (opposite the front
outward normal), with calibrated camera offset handled separately. Preserve
the angle interval width and provenance when transforming between frames.

Regression evidence should cover these 112 QR_002 tuples, a real two-stand
competitor, endpoint fragmentation, one missing beam, stale/moving witnesses,
and the seven-angle/one-center-sample Start transition. Retained-facing tests
must assert front-normal reversal, unchanged angular uncertainty, no refit,
center proof persistence, and independent route rejection.

The latest run's later return-to-Start route failure is separate from these
camera-admission failures and is outside this focused investigation.
