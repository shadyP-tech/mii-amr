# Opposite-side center handoff correction

Implements the correction identified in the `20260923T113013Z` admission audit.
The survey candidate stays immutable. Backside receipts now carry the current
three-scan association and, when available, the validated metric head-position
hypotheses and calibrated scan/camera transform. The position estimator checks
all hypotheses against that candidate's current scan and competing candidates.
Its uncertainty includes hypothesis spread, pixel-scale and model metrology
allowances; this is an engineering bound, not calibrated statistical coverage.
A scan-only estimate retains the larger surface-to-center uncertainty.

Frame projection carries this separate estimate with the original angle and its
uncertainty. Opposite planning uses it for the endpoint and terminal bearing,
checks the entire bounded angle at the endpoint, and includes an uncertain
current-target keepout while retaining the survey keepouts. Materialization and
route preflight independently bind it to the source receipt and snapshot. Arrival
admission uses the same projected estimate so it cannot undo the correction by
turning back toward the survey center. Live centering still uses current pixels,
range and the calibrated camera transform without another stand-angle fit.

Stopped reconciliation now runs before the ordinary/opposite observer split.
The opposite branch can use the authenticated retained center as its reference;
it still requires three fresh stopped scans, the original search/range envelope,
bounded displacement and competing-candidate exclusion. QR support must agree
with the resulting current cluster within the existing three-degree cone,
including finite-range parallax uncertainty. A projected background overlap
cannot truncate the foreground QR. Admission still requires an exclusive crop
and a fresh candidate-associated decode.

Centering support is independent of exclusive identity-crop permission. A turn
that cannot fit the existing 12-degree budget emits `centering_budget_exceeded`;
this and the opposite acquisition/conflict states feed the existing bounded
inspection progress/recovery path. A reconciled current scan can accumulate
unobservable-view progress even if no complete QR outline is available. This
progress grants neither identity nor motion. Successful opposite QR evidence
still takes precedence and has zero geometry grace.

The retained orientation cache rehashes its entire file dependency chain on each
hit; only unchanged historical validation is reused. Live scan/image checks are
never cached.

## Verification and limits

The regression fixture preserves the recorded backside proof, opposite tuples
000002–000004, frame 000004 JPEG, snapshots and frame projections. Tests cover:

- the previous stored-center rejection, current support and complete foreground crop;
- first decoded `Start` admission with the retained angle and no current fit;
- a sealed route using the measured center and its uncertainty;
- arrival avoiding a correction back toward the survey center;
- explicit out-of-budget recovery and freshness/epoch protections;
- changed history, metric position and retained-source rejection, including a warm cache.

The recorded measured center differs from the survey center by about 8.8 cm in
the arrival frame; its derived position allowance is about 2.7 cm. The 18-file
observer/navigation regression run passed 253 tests and 329 subtests, with one
WeChat-dependent test skipped. Additional arrival, position-tamper and crop-conflict recovery tests were
then added; the final recorded-failure suite passed all 10 tests.

Native OpenCV 5 decodes `Start` from the recorded search crop at 4x, but not from
this frame's rectified isolated quad. The producer-path admission regression
therefore supplies deterministic decoder output and separately checks saved-pixel
payload availability. End-to-end decoding and motion with the deployed WeChat
backend require a new real run. No robot motion or deployment was performed.
