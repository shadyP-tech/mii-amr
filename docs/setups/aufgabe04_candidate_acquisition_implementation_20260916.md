# Candidate head acquisition and independent identity processing

This change addresses the cluttered-window/radiator recording audited in
`aufgabe04_latest_viewer_head_audit_20260916T121146Z.md`. Physical geometry still
comes from the measured 3D head model and current image borders. QR size and
neck validation do not determine the angle.

## Acquisition and border selection

`stand_axis/head_search_bounds.py` applies candidate position and expected head
height before expensive hypothesis comparison. These are search constraints,
not measured corners. The observer uses its existing calibrated projection and
current scan association; it retains the full admitted search ROI so an
off-center complete head can be found before a pose exists.

`stand_axis/head_border_families.py` identifies repeated observations of the
same current supporting rails. Nearby physical edges are not grouped merely
because their rectangles have similar size. Separate stripe excursions and
independent nested frames remain distinct.

`stand_axis/head_proposal_selection.py` compares enclosing physical-frame
alternatives. Repeated small internal rectangles can explain composite texture
only when the finite supporting rail segments establish that topology. An
independent central rectangle is not discarded just because its coordinates
align with distant texture boxes. Unresolved independent heads and incomplete
comparisons remain unavailable.

`stand_axis/head_rail_intersections.py` recovers proposals from fragmented
current borders only when ordinary acquisition has found no verified head.
Four observed lines must have current side and corner support. This recovery
uses the remaining portion of the same 12 strict-verification budget; it cannot
reset the budget, replace an already verified ambiguous head, or accept a
partially compared winner.

Both the viewer and exploration use the shared cold acquisition and physical
fitting modules. Exploration no longer starts a generic locator and then a
second legacy locator on the same pixels. Current selected-border binding also
prevents the metric refinement from silently switching to another physical
border family after acquisition has compared the alternatives.

## Current pixels, identity and timing

Viewer tracking retains a search hint through at most two misses and for at
most two seconds. Its timestamp is not renewed on misses. Each new angle is
refitted on new pixels, and measurement freshness remains 250 ms. Missing
borders permit bounded reacquisition; an already verified ambiguous angle does
not trigger a search for a more convenient fit.

`observer/independent_qr_acquisition.py` runs physical geometry before optional
native/full identity work. The exact-image geometry holder permits one marker
decoration without a second acquisition or fit. A successful complete head can
receive its periodic identity probe immediately even when its crop was
recentered. A failed head acquisition can also receive a bounded identity
probe; repeated misses reserve up to 80 ms of the existing work allowance for
that probe. The sensor deadline is never extended.

The existing immediate-front policy joins one fresh accepted associated head
fit with a recent QR identity whose real corners lie inside that head. The
identity latch expires on motion, target/calibration/model changes, conflicts
or timeout. Text-only QR results remain insufficient. Backside appearance and
angle confidence remain separate; neither a failed decode nor skipped marker
work proves a backside.

`stand_axis/marker_work_schedule.py` separates optional marker checks from
geometry. Supplied current QR observations do not cause another native decode.
Failed physical acquisition skips optional native marker work. Complete
verified borders with an uncertain angle may still receive side checks.
Skipped or late checks mean **unknown side**.

The debug viewer's `--no-qr-decode` now disables both identity decoding and
hidden native marker processing. It remains a geometry-only mode and cannot
certify backside marker absence. Exploration uses the enabled marker policy
automatically; its existing mission command needs no additional flags.

Rectification maps are cached per processing stream and rebuilt when calibration
changes. Image pixels are always new. Viewer recordings now include all image
arrivals, including age-rejected arrivals, along with decode, rectification and
preparation timing. Header-to-receipt age is explicitly not labeled transport
latency because publisher/receiver clock offset has not been measured.

## Validation and limits

Validation covers background clutter, independent nearby rails, overlapping
printed rectangles, off-center heads, lost tracking, crop-adjusted intrinsics,
exact-image fit reuse, QR starvation, source-age overruns, identity conflicts,
backside marker uncertainty and the single-fit front artifact consumers.

Saved-image replay uses original recordings on mii001, a read-only repository
mount and an in-memory source overlay. Only derived JSON diagnostics return;
no recordings are copied and no ROS or robot motion is started. The replay
checks source hashes before and after execution.

Final focused regression run: **1,164 tests and 1,301 subtests passed, one test
skipped**, across 108 test files. `git diff --check` also passed. The tests cover
both recovery and negative cases: a missing border, independent nearby stripes,
multiple heads, exhausted budgets, moving/expired identity, and skipped QR work.

Final original-image replay of `recording_20260916_141146_741705309`:

| Mode | Accepted current geometry | Median processing | Accepted yaw range |
| --- | ---: | ---: | ---: |
| Unrestricted full-image cold acquisition | 0/20 | 175.2 ms | — |
| Independent acquisition with coarse candidate bounds | 12/20 | 89.8 ms | 15.17–15.30° |
| Same bounds with tracking and resets at frames 0, 8 and 15 | 19/20 | 20.7 ms | 15.26–15.63° |

The coarse replay bounds were image center `(477, 265)` and head height
`132 px`, without manually supplied corners. These were diagnostic priors,
not measurements from mission LiDAR. QR checks were disabled to isolate
geometry. Acquisition on frame 0 remained ambiguous; tracking acquired from
frame 1, and both later forced resets reacquired. Independent acquisition
retained five ambiguous frames and three other unavailable fits. Processing
times exclude transport and do not demonstrate live source freshness. These
results establish neither QR admission nor ground-truth physical angle accuracy.

Derived diagnostics, per-frame reasons/corners, source and source-overlay hashes,
and the replay body are saved in
[`candidate_acquisition_implementation`](../../results/aufgabe04/debug_audits/recording_20260916_141146_741705309/candidate_acquisition_implementation/replay_results.json).
The replay body runs under the in-memory source-overlay loader; the tested
stand-axis and observer source hashes match the final local implementation.
Original metadata and image hashes remained unchanged. No recordings were
copied, no remote checkout was modified, and no robot experiment was run.

The real viewer has no mission candidate projection before acquisition. Its
unconstrained full-image search therefore does not have exploration's measured
candidate bounds. A coarse image-location prior in a diagnostic replay is not
a replacement for real camera/LiDAR association. Fitting success and a stable
angle also do not independently establish which physical border is correct.

Live source freshness, publisher/receiver clock alignment, and uniquely
associated fresh samples still require hardware evidence. These changes do not
claim that all five candidates will be admitted at their first inspection pose.
