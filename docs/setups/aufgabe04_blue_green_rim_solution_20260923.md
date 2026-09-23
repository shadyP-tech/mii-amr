# Blue and green head ambiguity: investigation and proposed correction

Implementation follow-up: [production correction and validation](aufgabe04_blue_green_rim_implementation_20260923.md).
The report below records the preceding investigation and prototype.

## Conclusion

Use a coherent physical-rim locator inside the existing candidate search, then
fit and validate against the complete raw image. This resolves head acquisition
on the recorded blue/green cases in an offline prototype. Green still needs its
bounded-orientation policy: resolving its borders does not make a nearly frontal
planar pose a precise scalar angle.

This is an investigation, not a production change. The prototype uses each
recording's visually identified palette label. Automatic selection of a coherent
material component, fallback behavior, and end-to-end association/admission tests
remain implementation work. No robot motion or ROS nodes were started.

## Correcting the previous replay comparison

The earlier green fixture replay omitted `ImageSourceSupport.filter()`. The real
observer includes that filter. It rejects invalid rectification-canvas line
segments before the fixed rail quota, thereby changing which *valid* segments
reach pairing. In this frame it exposes more QR-texture combinations. This is
not a reason to disable source support.

With source support present, green again exhausts twelve verifications on both
local OpenCV 5.0.0 and ROS OpenCV 4.5.4. Thus OpenCV differences alone do not
explain the previous local success. The prior benchmark remains a measurement
of its narrower acquisition fixture, not a replay of the complete observer.

## Causes and rejected shortcuts

Blue's accepted proposals mix nearby outer-rim and inset border interpretations.
The initial call in `current_head_refinement.py` does not forward
`prefer_outer_metric_rail` into its initial raw fit, although the later outer
recovery receives it. Forwarding that preference consistently resolves this blue
frame, but introduces ambiguity on the simpler green replay. It is insufficient
as a general correction.

Green's alternatives include tall narrow rectangles assembled from QR texture.
In a diagnostic run with the verification limit raised to 64, 36–37 strict
checks completed and the result was still ambiguous. Increasing the limit
therefore adds work without resolving ownership of those borders. The production
limit was not changed.

Expanding small-contour texture hints and running the current repeated-texture
explanation earlier also failed to cover the competing rectangles and increased
work. The problem is not solved by caching or an arbitrary overlap threshold.

## Prototype and evidence

The successful sequence is:

1. Keep the original rectified image, source-domain filter, candidate bounds, and
   raw channel-union Canny edges.
2. Construct a separate locator edge view from existing Canny pixels near one
   coherent colored exterior. The HSV mask provides no measured edge pixels.
3. Acquire a complete head within the same bounds using that locator view.
4. Discard its measurement authority and independently refit all four borders
   using the complete original raw edge image.
5. Require `bind_selected_current_head` to confirm that this full-image fit
   preserves the selected current border family. Run the existing pose-quality
   and orientation-bound checks. Candidate/LiDAR association and QR identity
   checks remain downstream requirements.

Two representative frames were replayed on `mii001` inside the actual ROS
Apptainer image, using an explicitly approved temporary copy under
`/tmp/a04-border-investigation.EBMLG2`. The workstation checkout was not changed.

| ROS 4.5.4 replay | Acquisition before | Coherent-rim prototype | Full raw-image result |
| --- | --- | --- | --- |
| Blue / 000004 | Ambiguous, 7 strict checks | Selected, 1 strict check, 54 ms acquisition | Strict angle accepted; selected-border binding passes |
| Green / 000023 | Verification budget exceeded, 12 checks | Selected, 2 strict checks, 37 ms acquisition | Bounded orientation accepted; selected-border binding passes |

These timings are single offline acquisition measurements, not end-to-end ROS
latency. They exclude source preprocessing and the final full-image fit.

Local replay additionally covered every captured failure frame with complete
stored search metadata: **2 blue ambiguous frames and 10 green budget failures**.
All 12 acquired a head and preserved its border in the independent fit, using
1–2 strict checks. The other captures lack full search diagnostics after deadline
expiry and were not reconstructed with invented search inputs.

For green, all ten replays retained approximately **4.76° ±11.76°** in the
camera-relative orientation proof. They correctly retained
`head_model_yaw_uncertainty_too_high` for the stricter scalar-angle path. This is
orientation uncertainty, not remaining head-selection ambiguity. No ground-truth
angle measurement was available to claim improved absolute angle accuracy.

Synthetic probe controls kept two visible same-color heads ambiguous and rejected
both grayscale inputs and inputs with the upper rim removed. A pooled `all`
palette variant incorrectly found a blue proposal after the upper rim was
removed: paper/other palette support could substitute for the missing rim. That
variant is **not** the recommended production policy. Keep material components
separate and require a complete consistent rim; missing color evidence must
fall back to the ordinary detector, not silently suppress other heads.

Raw results and standalone probes, run from the repository root, are in
`results/implementation_checks/blue_green_solution_20260923/`:
`ros_rim_replay.json`, `local_rim_replay.json`, `sequence_replay.json`,
`negative_replay.json`, `coherent_rim_probe.py`, `sequence_probe.py`, and
`coherent_rim_negative_probe.py`.

## Recommended modular implementation

- Add a rim-locator module that identifies complete consistent material
  components automatically in the current bounded image. Color never denotes
  QR identity. Keep separate competing components and the source-support gate;
  do not choose a component because its color matches an expected station.
- Keep locator pixels separate from measurement pixels. A dedicated validator
  refits on the untouched raw image, binds the selected border, and carries
  full current-frame provenance. Do not reuse a proof hashed over masked edges
  as though it verified the full image.
- Account for the rim attempt and fallback inside one existing work deadline.
  Preserve ambiguity when independent heads remain. Explicit tests must cover
  differently colored/gray neighboring heads, weak or broken rims, and all
  physical stand colors; the small prototype controls do not prove those cases.
- Preserve green's complete orientation interval. The existing bounded producer
  and artifact validator require at least seven current samples. Faster
  acquisition can help collect them, but immediate QR-only exit can precede
  that window. Store valid geometry evidence separately from successful QR
  identity. If a one-frame bounded-front facing receipt is desired, give it an
  explicit independently validated policy; never relabel one frame as temporal
  consensus or lower the existing seven-sample contract globally.
- Validate any facing endpoint for the entire interval, current center
  uncertainty, and route clearance. If that cannot pass, request a bounded
  off-axis inspection only when facing precision is actually required. Preserve
  successful QR admission. This does not justify refitting an already retained
  backside angle on the opposite-side identity branch.

## Retained-angle logistics catalog limitation

The separate catalog problem is confirmed in
`stations/autonomous_arrival_catalog.py`: its admission policy accepts a strict
current-head QR receipt or seven-frame QR consensus, and thus rejects schema-4
retained-angle receipts. Simply adding schema 4 to the accepted policy is not
sufficient. `_project_recommendation` changes the observation frame, whereas
schema-4 source validation binds the unchanged original QR receipt. The catalog
also requires the recommendation center to equal the frozen candidate center,
which intentionally differs from a validated measured center in this path.

Implement a separate authenticated projection wrapper: preserve the original
QR/angle/center receipt, reference both frame certificates, transform the center,
full angle interval and endpoint, and validate the derivation. Recheck clearance
against both the original candidate keepout and measured-center envelope in the
common frame. Only then can catalog promotion accept the retained-angle policy.
This is independent of blue/green image acquisition and needs its own regression
tests before logistics use.
