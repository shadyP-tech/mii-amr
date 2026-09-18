# Backside branch audit — run 20260918T143813Z

The camera repeatedly detected the light-grey backside head. The main blocker
was scan-boundary fragmentation in the camera/LiDAR association, which prevented
the observer from committing a certified backside-axis receipt. The opposite-side
branch requires that receipt; without it, the candidate became `unobservable`
at the deadline and the planner selected a generic 90-degree inspection view.

## Scope and evidence

- Latest workstation run at inspection: `stand_explore_exact2_camera_all5_20260918T143813Z`.
- Recorded revision: `74726fd56dc960d1bcef5d7f81e5eaa9d88d95a9`, clean recorded checkout.
- Candidate: `survey_candidate_0001`; camera shows the plain light-grey backside
  with a handwritten label. No QR identity was established for it.
- Saved mission and parent bundle copied read-only from `mii001` into the same
  repository-relative `results/` paths locally. No robot commands were issued.
- Analysis/replay: `results/aufgabe04/implementation_checks/backside_branch_audit_20260918T143813Z/analyze.py`.
- Derived counts, per-frame source hashes, and raw-scan replay evidence:
  `results/aufgabe04/implementation_checks/backside_branch_audit_20260918T143813Z/evidence_summary.json`.

The saved parent log ends at the generic inspection leg's diagnostic collection
at 14:46:12 UTC; a second read returned the same ending. This audit establishes
why the first backside view did not trigger the certified opposite branch. It
does not establish completion or execution of the later generic route.

## What worked

The previous missing-centering-module crash is resolved in this run. Both child
turns completed, with measured angular travel of 0.101424 and 0.040605 radians
(5.811 and 2.326 degrees), totaling 8.138 degrees. The subsequent image places
the actual head near horizontal pixel 400 in the 800-pixel image.

The final post-centering capture contains 203 saved frames, with no capture
drops. Three failed TF acquisition; 200 reached the detector. Classification
of those detector frames:

| Recorded result | Frames |
| --- | ---: |
| Current head geometry and pose quality accepted | 166 |
| Of those, rejected for ambiguous current LiDAR clusters | 149 |
| Of those, unique current LiDAR association accepted | 17 |
| Backside axis samples among the 17 | 16 |
| Other axis sample among the 17 | 1 |
| Geometry/acquisition failures | 34 |

The 34 geometry failures comprise 28 unavailable-border results, five acquisition
deadlines, and one ambiguous head proposal. Geometry still has room to improve,
but it is not the dominant reason this branch failed.

Frame 50 is a concrete rejected example: the original image clearly contains
the complete backside head, and recorded `head_model_quality.accepted` is true.
Minimum measured edge length is 100 px, raw border support is 1.0, reprojection
RMSE is 0.249 px, and estimated yaw standard deviation is 1.748 degrees. The
backside appearance check also passes. The later unique-association check fails.

## Main cause: target straddles the scan boundary

After centering, the camera-registered target bearing is about -0.54 degrees.
Its returns occur at both the last indices and the first indices of the
full-rotation scan. Joining these endpoints requires the `ScanTopology`
validation to accept the original angular metadata.

For all 149 rejected good-head frames, recorded association is
`ambiguous_registered_camera_clusters`: two groups rather than a unique target.
Topology rejected endpoint adjacency for:

- 107 frames: `inconsistent_scan_endpoint_metadata`.
- 42 frames: `seam_is_not_one_sampling_step`.

The raw scans were replayed through the unchanged association functions, using
their recorded bearings, range interval, sensor time, and spatial thresholds.
All 166 results matched the recording. Every one of the 149 ambiguous cases
consists of the two endpoint groups; the endpoint pair passes both the existing
4-cm point-gap limit and 5-cm range-jump limit. These measurements support the
scan-boundary diagnosis, but do not independently validate changing the scanner's
angular contract or authorizing a merge.

For example, frame 50 has 224 beams and target returns at indices 0, 1, and 223:
approximately 0.058, 1.658, and 356.857 degrees, at ranges 0.543, 0.547, and
0.539 m. Its indexed seam gap is 2.000 sampling steps. Its reported `angle_max`
disagrees with the indexed last angle by 0.965 steps, exceeding the allowed
0.25-step endpoint error. The existing conservative policy therefore keeps
the endpoint groups separate despite their close physical spacing.

By comparison, frame 93 has valid full-rotation metadata and is accepted as
one cluster with source indices `[221, 222, 223, 0]`. Its backside head crop
and axis sample are accepted.

The temporal scan-fragment recovery cannot bridge this case: it explicitly
supports one missing *internal* beam and rejects boundary-crossing witnesses.
Its repeated reason is `current fragments are not separated by one missing internal beam`.

## Why this appeared as head-frame failure

The current pipeline ties backside usability to complete-head registration.
When current LiDAR association is ambiguous, registration reports
`complete_head_unique_association_required`. The backside crop gate then replaces
the usable pose result with `backside_complete_head_crop_unverified` and an
unobservable state. This happened in 139 frames; ten other good-head frames
reported `lidar_target_mismatch`. Thus the final status hides a successful
visual head measurement behind a later target-association failure.

The map projection is also displaced from the measured head: about u=496–497
versus u=400, a camera/map bearing difference around 7.8 degrees. The original
map-centered 3-degree cone misses the stand. The explicitly bounded camera
registration correctly shifts the search within its 12-degree allowance;
the remaining failure is uniqueness at the scan seam, not a range-gate failure.
The recording alone cannot assign that projection error to localization,
candidate-center error, or camera calibration.

## Why the opposite-side branch did not run

The configured observer requires seven same-source fresh axis samples within
a five-second TTL, plus current backside evidence. After the two centering
turns, accepted backside samples were sparse: 16 over about 44 seconds. The
window peaked at six and ended with five. The evidence was not poisoned and
there was no motion-epoch reset, but older samples expired before seven could
accumulate. No `axis_observation.json` was committed.

The candidate inspection loop invokes `move_opposite` only when
`observation.axis_observation_path` exists. The final observer instead timed out
and was recorded as `observation_unavailable`, classification `unobservable`.
Its achieved canonical view normal was 0.244381 rad; the proposed generic view
was 1.815177 rad, exactly +90 degrees. This follows the `unobservable` search
policy. It is not a failed opposite-side route admission: that route branch
was never entered.

TF delays were a secondary cost: the first post-turn capture had 28 TF failures
before its next centering advisory. Centering, restart/arrival work, and those
delays shared the 90-second physical-view deadline, leaving roughly 45 seconds
for the final observer. Nevertheless, its 200 processed frames already show
that scan association, rather than simply insufficient image opportunities,
is the principal bottleneck.

## Recommended correction

1. Establish the actual LDS scan endpoint contract from recorded/raw driver
   evidence. Correct inconsistent metadata at its source if possible, or add
   an explicit, tightly bounded endpoint-fragment association policy supported
   by independent scan evidence. Preserve range, point-gap, competing-target,
   freshness, and current-camera binding checks. Do not blindly make every
   near-360-degree scan circular.
2. Add recorded-scan regressions for both frame 50's rejected two-step seam and
   frame 93's valid one-step seam, plus genuinely distinct targets and partial
   scans. The present offline replay establishes a reproducible baseline.
3. Separate `head_detected`/pose quality from `target_association_accepted` in
   observer diagnostics so a LiDAR veto is not presented as missing head geometry.
4. Reassess TF restart latency and timeout allocation afterward. Increasing
   the timeout or reducing seven samples to six does not address the recurring
   scan-boundary rejection and should not substitute for repairing association.

No production code was changed by this audit. Fixing the association bottleneck
is necessary; a new replay/end-to-end check is still required before claiming
that a corrected implementation will emit the backside receipt and admit a
physical opposite-side route.

## Implemented correction and offline validation

The subsequent requested fix adds a separately identified
`scan_endpoint_fragments_witnessed` proof to stopped-scan persistence. It does
not change `ScanTopology` or declare malformed scans circular. Endpoint recovery
is limited to an explicitly full-rotation scan with at most a two-step indexed
seam (bounded tolerance), a near-one-step reported seam, bounded endpoint
metadata error, and no more than two degrees per beam.

Three distinct earlier scans must each independently contain one contiguous
target under the original topology checks. Each must supply real connecting
returns and support both current fragments at the same stopped candidate pose.
The existing 2.5-second history, one-second maximum inter-scan gap, 0.5-second
source freshness, range gates, 4-cm spatial gate, and 5-cm range-jump gate remain.
The current raw cluster count stays two in the receipt. Resolved fragments do
not become witnesses or extend their lifetime. Partial compatibility checks
may preserve pending witnesses but cannot return an accepted target.

The observer now records visual head detection and conditional pose quality
before the backside association gate, alongside the separate target-association
result and reason. This prevents a LiDAR rejection from hiding a detected head
in the detailed diagnostics.

Recorded evidence verification:

- Original raw association accepts 17 of the 166 quality-admitted head frames.
  The bounded correction accepts 31, recovering 14 endpoint cases; it continues
  to reject cases without enough recent independent supporting scans.
- Recorded frames 52, 54 and 56 supply the three real witnesses. Frames 58–61
  then supply four additional accepted backside observations. Frame 57 has a
  possible marker and does not count as backside evidence.
- The recorded angles, corners, exact poses, scan/image times, and appearance
  measurements pass the existing temporal border/angle checks and backside
  confidence accumulator: seven samples within five seconds by frame 61
  (14:45:26.863 UTC image time). The same sequence without the correction has
  only three accepted backside samples.
- A separate observer integration test runs the actual evidence, receipt
  validation and candidate inspection loop with simulated sensor transport and
  pixel fitting. It commits the witnessed-endpoint backside-axis receipt and
  calls the opposite-side branch before generic view proposals.
- Negative tests reject insufficient/duplicate witnesses, wrong candidate or
  epoch, motion, stale evidence, malformed scan endpoints, additional targets,
  missing endpoints, spatial contradictions, and modified persisted proofs.
  Preview operations cannot mutate live evidence.

The new recorded regression fixture is
`tests/aufgabe04/fixtures/scan_endpoint_20260918.json`; source frame metadata
hashes are included. Full recording association replay and results are in
`results/aufgabe04/implementation_checks/backside_branch_audit_20260918T143813Z/replay_fix.py`
and `fix_replay_results.json`.

The broader 69-file regression selection completed with **776 passed, 823 passing
subtests and two failures**. Both failures were reproduced independently on an
unchanged export of revision `74726fd`: QR ID casing in
`test_accepts_valid_qr_id_and_builds_row`, and the missing
`odom_execution_certificate_sha256` fixture in
`test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`.
They are outside this correction.

These checks reuse the recorded visual measurements; they are not a new full
camera-pixel replay or physical route execution. No workstation deployment or
robot motion was performed. The existing full-rotation run command needs no new
flags once the workstation receives the updated code.
