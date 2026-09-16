# Bounded head admission — implementation and validation

This implements the correction identified in
[the 10:35 UTC run audit](aufgabe04_candidate_backside_front_audit_20260916T103525Z.md).
The complete current 3D head fit can now supply an orientation interval even
when a unique angle fails the existing ambiguity or 3° uncertainty gate.
It does not mark that rejected single-angle estimate as usable.

## Module boundaries

- `perception/stand_axis/head_orientation_bounds.py` retains every plausible
  IPPE pose within the current residual neighborhood, computes each pose's
  pixel-noise uncertainty and encloses all poses with a three-sigma engineering
  allowance. Current corners, physical profile, raw borders, completeness,
  positive depth and numerical conditioning remain mandatory.
- `real_robot/observer/current_head_detection.py` and the association/tracking
  adapters bind those pixels to their current crop-adjusted camera and unique
  LiDAR target. A tracking pose remains only a search locator.
- `real_robot/observer/bounded_head_window.py` retains at least seven distinct
  ordinary fresh, stopped, associated samples. It encloses every interval
  without averaging away uncertainty and separately preserves the existing
  0.04 normalized corner-displacement limit. Incompatible samples remain until
  bounded age/capacity eviction. Epoch, target, calibration and face changes
  cannot share a window.
- `real_robot/observer/bounded_head_observation.py` binds front identity or
  repeated backside appearance independently to the interval. Current and
  historical front-marker evidence still veto backside commitment. A complete
  bounded receipt takes precedence over the older single-angle writer.
- `artifacts/bounded_orientation.py` owns the shared interval contract and
  analytic endpoint check. Existing circular stand keepouts and static-map
  checks validate the same proposed route. Endpoint checking includes grid
  offset, 3 cm terminal position tolerance and candidate-center uncertainty.
  Every plausible angle must preserve the selected face and useful viewing
  incidence. The current conservative policy caps half-width at 15° and worst
  viewing incidence at 20°; these are limits on the retained range and planned
  view, not a ±15° limit on the measured stand orientation.

All module paths above are relative to `scripts/aufgabe04/`.

Bounded front recommendations use schema 2. Schema 1 forbids bounds; schema 2
requires them. Frame reprojection rotates interval centers and preserves their
width. Unsupported legacy planners/converters reject intervals explicitly.
Backside receipts retain their existing sensor/appearance proofs, with interval
validation replacing the single-angle confidence minimum for that variant.
Their point-angle confidence is deliberately zero rather than manufactured.

## Recovery behavior

Transient misses preserve an already established stopped front-recovery
deadline, without adding evidence or extending its 30-second budget. Identity
conflicts and motion still invalidate the observation epoch. Legacy backside
consensus now checks the receipt's actual confidence minimum before declaring
it ready. Intervals cannot be silently discarded through a legacy success path.

Fresh, associated bounded-head hints can prioritize a ±20° inspection adjustment
when the interval/view is unsuitable. This is an advisory search direction;
ordinary collision, localization and execution admission still apply. It does
not declare an exact normal or authorize a route by itself.

## Validation and limits

Regression tests cover ambiguous current-pixel fits, noise expansion and axial
wrap; crop/calibration/scan binding; seven-frame front and backside receipts;
stale data, motion, conflicting QR identities and front-marker vetoes; temporal
border changes; schema downgrade/stripping; interval-preserving frame changes;
and one actual route endpoint checked across all retained angles. Recovery tests
reproduce the premature exit with 18 seconds still available.

Final related-suite validation used
`/tmp/a04-head-selection-venv/bin/python -m pytest` across 60 perception,
observer, artifact and candidate-planning modules: **619 tests and 769 subtests
passed**. After adding the last two endpoint/completion regressions, the two
affected observation/adjustment modules were rerun: **16 tests and 19 subtests
passed**. `git diff --check` also passed.

The covariance allowance is an engineering pixel-noise model, **not calibrated
physical angle accuracy**. This change does not claim that every clearly visible
head must pass, or that the latest real run will necessarily succeed. Border
instability and excessive uncertainty still require another view. No code,
images or recordings were transferred to the workstation, and no robot action
was performed while implementing this correction. Hardware validation remains
the next evidence step.
