# Improving candidate-local LiDAR scan support

Investigation date: September 30, 2026  
Recorded run: `stand_explore_exact2_camera_all5_20260930T112749Z`  
Candidate: `survey_candidate_0003`

## Conclusion

Improve acquisition and observability before relaxing the fitter. The immediate
priorities are a longer fixed stopped cohort, original scan metadata, deliberate
placement away from the scan boundary, and near-normal viewpoints with enough
predicted beams. These address different failure modes. No single clustering
threshold change can repair this run.

Camera acquisition should remain available at the first admitted arrival while
LiDAR orientation is unresolved, as described in the companion run audit.
Better LiDAR support must not reinstate a blocking series of unsupported orbits.

## What the saved scans establish

The local scan spacing is approximately 1.63 degrees, with inter-scan timestamps
approximately 0.095 seconds apart. Each acquisition takes exactly three scans,
spanning 0.19–0.20 seconds. Their results remain 2/3, 0/3 and 0/3 accepted fits.

- `view_00`: accepted scans contain five and four contiguous returns. The last
  scan contains `[0, 1, 3, 4]`; raw bin 2 is invalid, not merely filtered out by
  the candidate envelope. The two fragments are rejected. Joining them without
  evidence would also be insufficient: their combined points fail the existing
  surface fitter in this recorded scan.
- `view_02`: target points occupy the beginning and end of the scan. The first
  scan has `[0, 1, 2, 3, 219]`, the next `[0, 1, 2, 219]`, and the last
  `[0, 2, 3, 219]`. The last also has an invalid internal bin 1.
- `view_04`: the first two scans have only three target returns. The last has
  four points split across `[0, 1, 2, 219]`.

The first/last angular gaps inferred from the indexed geometry vary from
approximately 1.15 to 2.69 nominal steps over the nine scans. The original
`angle_max` is missing, so these gaps cannot be authenticated as an ordinary
one-step circular seam. The current `ScanTopology` contract must not be bypassed
by fabricating `angle_max` or changing the increment to make a full circle.

The first support move adds almost no independent bearing: only 0.75 cm net base
displacement and 0.34 degrees scanner-viewpoint change. The second changes the
viewpoint by approximately 56.8 degrees but still lacks sufficient returns.
There is no saved physical head-angle ground truth or camera observation.

## Missing beams: collect useful temporal support

The current three-scan contract was designed for stopped tour/occupancy evidence.
`tour_scan_capture.py` stops on the first valid three-scan cohort, regardless of
whether those scans contain enough head returns. The geometric fitter requires
at least three accepted scans and 75% support, so its effective three-scan
requirement is 100% acceptance.

| Fixed cohort size | Accepted fits required | Rejected scans tolerated |
| --- | --- | --- |
| 3 | 3 | 0 |
| 4 | 3 | 1 |
| 6 | 5 | 1 |
| 8 | 6 | 2 |
| 10 | 8 | 2 |

A candidate-specific fixed eight-scan window is a reasonable experiment at the
recorded scan rate: approximately 0.67 seconds between its first and last scan.
Keep its overall collection deadline finite, and explicitly bound the stationary
history window, for example to one second at this observed rate. A four-scan
cohort is a smaller change that already tolerates one rejected scan. Neither
choice establishes a measured success rate from these three-scan recordings.

Implement this as a candidate-head capture contract rather than globally
changing the tour capture's exact-three/.5-second rules. Retain exact source-time
scanner and base transforms, source order, duplicate rejection, stationary
pose/mount checks, candidate binding, and fresh last supporting scan. The arrival
verifier already distinguishes bounded history from current source age, but the
capture adapter and persisted schema require coordinated updates.

Keep every examined scan in the 75% denominator, including scans with no usable
fit. Do not keep trying until three favorable scans can be selected and hide the
failed scans. More frames do not narrow systematic geometric uncertainty or
create a second independent viewpoint. Existing tangent consistency and center
interval agreement must still pass; a longer cohort can reveal disagreement.

The camera association path already has a narrower mechanism in
`real_robot/observer/scan_target_persistence.py`: three earlier, independently
valid stopped scans can witness a bounded current missing interval. It preserves
the current fragments and requires real historical returns across the gap.
This is a useful pattern for identity continuity, but it does not certify a
head normal, and should not simply turn a fragmented scan into an accepted
orientation fit. The current first cohort has only two accepted surface fits.

## Scan boundary: preserve metadata and avoid the gap physically

Retain original `angle_min`, `angle_max`, `angle_increment`, sample count,
`scan_time`, `time_increment`, topology profile, and source stamps. Keep
intensities/confidence and invalid-range reason categories in diagnostic evidence.
The present cohort and receipt schemas retain neither endpoint metadata nor
intensity, and collapse different invalid conditions into `None`.

Use two distinct cases:

1. If original metadata proves a one-step full-rotation seam, the existing
   topology helper can permit endpoint adjacency. Apply the actual angular gap,
   spatial gap, depth discontinuity, compactness and competing-candidate gates
   before fitting. Preserve the original indices and do not invent a return.
2. If the seam contains a genuine missing interval or inconsistent metadata,
   keep the fragments separate. Collect more stopped scans or move the target
   away from the boundary. A witness-backed fragment identity proof is different
   from a continuous surface observation.

At every recorded local stop the head occupies indices near zero. The generic
probe terminal yaw points the base toward the candidate, repeatedly placing the
target near the forward scan boundary. Therefore, orbiting while keeping the
same terminal-facing rule repeats the boundary problem at the new location.

Before translating, evaluate a small sealed yaw change that places the complete
candidate angular envelope away from both scan endpoints, with margin for the
observed boundary gap, angle-min variation, candidate uncertainty and pose error.
A roughly 15–25 degree yaw adjustment is a starting experiment, not a universal
constant. Compute its actual direction from scan geometry and check the existing
turn/clearance gates. Capture after stopping, start a new stopped epoch, then
return to calibrated camera facing before the camera observation if required.
Yaw improves scan phase/boundary placement; it does not improve physical face
incidence or count as an independent 20-degree viewpoint baseline.

Do not assume this yaw action fixes all failures. It cannot manufacture returns
from an edge-on head, and the first recorded dropout remains an internal gap.

## Too few returns: plan for angular width and face incidence

The measured head width is 7.8 cm. A noise-free ray/phase sweep uses the recorded
median 1.630-degree increment and the production surface fitter. Distances here
are scanner-to-head-center distances, not the robot-base approach offset.

| Distance | Incidence from head normal | Ideal returns over sampling phase |
| --- | --- | --- |
| 0.50 m | 0 degrees | 5–6 |
| 0.55 m | 0 degrees | 5–6 |
| 0.55 m | 30 degrees | 4–5 |
| 0.55 m | 45 degrees | 3–4 |
| 0.55 m | 60 degrees | 2–3 |
| 0.65 m | 0 degrees | 4–5 |
| 0.65 m | 30 degrees | 3–4 |
| 1.50 m | 0 degrees | 1–2 |

These ideal counts exclude dropout, occlusion, driver filtering and binning.
The phase-fit fraction in the JSON is a sensitivity result, not a physical
success probability. Actual returns may be lower, as the recordings show.

Use predicted angular support to rank proposed acquisition poses. Prefer a
near-normal face view with five or more expected real returns, enough span for
the fit, route clearance, and scanner-height observability. Moving from 0.55 to
0.65 m can worsen support. Requesting a closer base standoff must pass calibrated
scanner geometry, original keepouts and all uncertainty/clearance checks.

Once a usable single-view tangent or camera head-angle estimate is available,
choose the second separated viewpoint near its normal, rather than using a
blind +60-degree increment. Two views around the normal, separated by at least
20 degrees modulo pi, can provide both broad-face support and independent
geometry. If orientation is unknown, camera evidence can guide the choice;
an arbitrary radial direction must not be labeled a measured head normal.

## Driver diagnostics and measurement uncertainty

The upstream ROBOTIS Humble driver assembles a revolution, transforms/filters
the points, fills a uniform angular grid with NaN, and places points into integer
bins. Multiple points in a bin retain the shortest range. This can produce
unfilled bins even when the sensor supplied physical points. Its endpoint and
indexed geometry are also distinct. See
[upstream conversion](https://github.com/ROBOTIS-GIT/ld08_driver/blob/humble/src/lipkg.cpp).
This source was inspected as an explanation to test; the exact deployed driver
revision has not been established by this investigation.

The upstream near filter processes close-range data before publication. Record
the installed version, raw/filtered point counts, packet/CRC failures, confidence
and bin occupancy before attributing an invalid bin solely to reflectivity or
sensor loss. Compare raw and filtered data passively on a diagnostic path;
changing the navigation scan filter globally is not the first correction.

The manufacturer specifies fixed 2.3 kHz sampling and notes that angular
resolution varies with scan frequency. Publishing an interpolated 360-bin scan
would not create new measurements. Physical scan-frequency control is a hardware
question; the inspected publisher exposes frame/namespace parameters, not a
resolution setting. See [LDS-02 specification](https://emanual.robotis.com/docs/en/platform/turtlebot3/appendix_lds_02/).

The existing fitter's 3 mm point-noise allowance is an uncalibrated engineering
assumption. The manufacturer's absolute range accuracy is not a validation of
that allowance. Driver binning can also introduce angular quantization. Measure
range repeatability, intra-head residuals, angular bias and actual normal error
at known poses before making calibrated angle-accuracy claims. Repeated scans
cannot average away fixed range bias or binning error.

## Suggested implementation order and validation

1. Record complete candidate-local scan metadata and per-scan support diagnostics.
   Add a candidate-specific fixed cohort with a finite deadline and unchanged
   fit fraction, spatial support, ambiguity and freshness requirements.
2. Keep first-arrival camera acquisition responsive. Use an optional scan-aware
   yaw recovery before a large orbit when boundary fragmentation is diagnosed.
3. Rank translation probes by expected beam count, face incidence, independent
   viewpoint separation and clearance. Reject negligible support improvements.
4. Add topology-aware fitting only with actual original metadata. Consider
   witness-backed identity continuity separately from angle verification.

Offline regressions should cover intermittent internal invalid bins, authentic
one-step seams, wider/inconsistent seams, competing objects, insufficient beam
count, duplicate/stale sources, motion during collection, unchanged uncertainty
under repetition, and guaranteed camera handoff. Real validation should compare
fixed cohorts and a small yaw adjustment at the same location, then compare
near-normal separated views. Record fit fractions and measured head-angle errors;
do not infer success from a pose command or a synthetic sampling count.

## Reproduction

The saved replay is `results/implementation_checks/run_audit_20260930T112749Z/lidar_replay.py`.
The additional reproducible investigation is:

- `results/implementation_checks/run_audit_20260930T112749Z/lidar_support_study.py`
- `results/implementation_checks/run_audit_20260930T112749Z/lidar_support_study.json`

The latter records per-scan invalid bins and missing metadata, source hashes,
cohort support arithmetic and the idealized phase sweep. It uses the actual
production geometry fitter without changing admission. The initial investigation did not modify production code or driver configuration,
or execute robot motion. The implementation below was added afterward.


## Implemented correction

Candidate-local head capture now collects exactly eight distinct source-stamped
scans within 1.5 seconds, with a three-second acquisition deadline. The shared
stationarity, exact-time transform, per-receipt freshness and fixed scanner mount
checks remain in force. Stored-tour capture retains its three-scan contract;
legacy candidate artifacts remain readable. Fit admission retains the 75% usable
scan requirement, four real adjacent returns per fitted scan, geometric checks,
competing-candidate rejection and independent viewpoints. Six passing scans out
of eight can qualify; five cannot.

New candidate receipts retain original `angle_max`, timing, intensity values and
per-bin invalid-range reasons in content-hashed schema 3. Per-scan support
artifacts show original indices, cluster indices, topology decisions and fit
rejection reasons. Endpoint groups join only when the declared full-rotation
profile and original geometry prove a one-sampling-step seam. Wider seams and
internal missing bins remain fragmented; no synthetic rays are generated.

When a majority of scans exhibit boundary fragmentation and no independent
normal hint is available, optional recovery proposes one stationary sampling
turn per candidate. It projects the whole candidate envelope through the actual
scanner translation and yaw into the original scan interval, with a beam and
pose margin, and selects a useful turn between 15 and 25 degrees. If no bounded
turn clears the envelope, it rejects the proposal before motion. The turn uses
the existing sealed child, single-use claim, exclusive velocity publisher,
fresh live candidate support, odometry, clearance and stop checks. Translation
is zero; measured angular travel is limited to 26 degrees. Its purpose is
`candidate_lidar_sampling`, and it does not claim camera centering, head alignment
or an independent viewpoint. The expanded mission RUN scope explicitly includes
this action; older scopes retain their earlier permissions but cannot authorize
sampling. No extra operator prompt or command-line option is added.

Support routes prefer close feasible standoffs starting at the configured
approach offset, bounded to 0.50–0.55 m, with two farther alternatives at 5 cm
increments. With a usable single-view fit, probe directions use the measured
normal and separated ±25-degree views. Predictions account for face incidence,
actual angular spacing, scanner translation and the materialized route goal.
A fitted support goal predicting fewer than four returns is rejected before
motion. Predictions are noise-free diagnostics, not confidence estimates;
post-arrival physical scans still determine whether a hint or alignment exists.

First-arrival camera capture remains first. Optional recovery yields back to
camera after at most one successful move, including a sampling turn. Sampling,
probe, alignment and elapsed-time budgets persist across camera attempts. Each
source scan cohort binds to the current projected candidate snapshot, including
localization frame changes.

No robot motion or driver reconfiguration was performed to validate this patch.
Physical success and angle accuracy still require a new logged robot run.


## Verification of the implementation

The focused capture, metadata, fitting, motion permit/child/runtime, planning,
arrival, candidate inspection and opposite-side regression suite passed 299
tests and 202 subtests. This includes 17 new scan-support tests and a regression
for the configured 0.50 m support probe. Its JUnit receipt is saved under
`results/implementation_checks/lidar_scan_support_recovery_20260930/targeted_tests.xml`.
Compilation and `git diff --check` pass.

The broader Aufgabe 04 run reported 4,695 passes, 74 failures, two skips and
3,820 passing subtests. Many runner failures were caused by the sandbox denying
Unix socket creation. Re-running those five runner/preflight test files with
local socket access produced 87 passes, 20 passing subtests and one failure:
`test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`
uses a certificate fixture missing `odom_execution_certificate_sha256`.
Other broad-suite failures include camera/QR expectations, route-diagnostics
fixtures and stale module inventories. The full suite is therefore not green;
no full baseline comparison was performed and these results do not certify a
physical robot run. The focused tests exercise the implemented LiDAR changes
without ROS motion.
