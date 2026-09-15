# Current head geometry admission correction — 2026-09-14

The measured 3D head remains the sole angle source. QR decoding and finder
checks supply identity and front/back evidence; QR corners do not supply a
replacement head angle. This change addresses the boundary-selection and
temporal-window defects identified in the camera admission audit.

## Physical boundary selection

`perception/stand_axis/head_outer_border.py` owns bounded raw-border recovery
and the shared `current_head_boundary_eligible` contract.

- Preserve the original current proposal through raw refinement. If a wide
  search shifts inward beyond the existing two-pixel rail allowance plus
  profile metrology tolerance, try a narrow current-pixel fit at that proposal.
- Retain the two bounded outward searches based on measured panel/symbol size.
  There are at most four strict refinements including the initial fit. A
  proposal only locates the search; all accepted corners require current raw
  borders and corner arms.
- An enclosing alternative must have meaningful separation in both image
  dimensions before it can independently resolve a QR-size disagreement. The
  minimum growth comes from measured head size divided by paper size plus
  recorded tolerance: `.078 / (.071 + .002)`, approximately `1.06849` for this
  profile. This prevents the two edges of a thick paper stroke from being
  counted as independent physical head recovery.
- QR/head span disagreement is diagnostic for such a recovered outer frame.
  A lone disputed quadrilateral remains unresolved and requests reacquisition.
  Neither low reprojection error nor absence of decoded QR proves physical scale.

`head_model_fit.py` fits the selected current head. `model_pipeline.py` combines
its boundary evidence with independent QR diagnostics and preserves the
original acquisition failure reason when no raw head was available.

The shared boundary contract binds the selected corners, original proposal,
current alternatives, measured head dimensions and profile. Measured-angle
admission, backside appearance/classification and the temporal adapter all use
this contract. Neck validation remains unnecessary.

## Temporal consistency and side evidence

`real_robot/observer/head_observation_window.py` admits only verified current
head fits or explicit planar ambiguity of a verified physical head. A retained
pose hypothesis of a rejected paper/non-head boundary is diagnostic only and
cannot clear earlier head samples. Rejected frames add no axis evidence and
cannot complete an observation.

Genuine competing orientations of the same physical head still veto old
angles. Their residual neighborhood now matches the quality producer's
existing pixel-noise bound, including its 0.75-pixel floor. This does not
authorize an ambiguous angle or backfill previous measurements.

Seven fresh, consistent, associated samples remain required. Freshness,
positive depth, reprojection error, yaw uncertainty, candidate/LiDAR binding,
QR identity binding, stopped-pose checks and motion admission are unchanged.
A current verified marker on a rejected frame still vetoes the backside epoch.

## Viewer

`perception/debug/viewer_model_overlay_policy.py` shares measured-head admission
with the observer. Purple dashed geometry indicates a fresh admitted
single-frame fit. Rejected or obsolete geometry is gray, and locator proposals
are amber and labeled. Freshness is checked again at rendering and recorded in
the viewer metadata. Single-frame fit status is separate from handoff readiness.
Validation uses crop-local geometry; rendering alone applies image offsets.

## Validation and limitations

Final focused run: **296 tests and 583 subtests passed** in 8.72 seconds.
All 19 changed Python files parse with Python 3.10 syntax; `git diff --check`
passes. This is a focused perception/observer/viewer run, not the full repository
test suite.

Focused offline validation covers:

- Recovery of a nine-pixel inward border switch from current outer pixels.
- Identical head corners and angle when only the QR span diagnostic changes.
- A 96-case paper-only sweep across angle, distance and stroke thickness; all
  33 usable fits with contradictory QR scale remain unresolved.
- Exact boundary/profile/proposal binding across the four consumers.
- Six valid samples, then a rejected panel, then a new valid seventh sample:
  both front-facing completion and backside receipt generation succeed without
  counting the rejected frame. A genuine unstable head still clears old angles.
- The actual observer's seven-sample backside receipt selects the opposite-side
  branch before generic inspection views. These integration tests inject image
  fits and ROS transport; they are not physical robot experiments.
- Existing local recorded-image fixtures, angle-reference tests through 75
  degrees, freshness/association rejection and viewer rendering checks.

Testing uses local Python 3.12 and OpenCV 4.13; changed sources are also checked
for Python 3.10 syntax. No current recording or code was transferred, and no ROS
or robot motion was run. Existing local recorded fixtures are not a replay of
the newest run. Hardware throughput and first-view success still need a real
experiment. This correction does not repair the separately audited drift of
stored candidate coordinates across localization updates.
