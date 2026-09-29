# LiDAR cluster stability during camera exploration — 2026-09-29

Successful QR decoding does not by itself associate the symbol with the current
candidate. The observer also requires the symbol's calibrated image ray to agree
with fresh LiDAR support, the admitted candidate range and the applicable
candidate-registration proof. This investigation found two remaining history
defects that unnecessarily discard valid support at stopped inspection poses.

## Findings and recorded evidence

1. **Internal dropouts erased partial witness history.**
   `observer/scan_target_persistence.py` already validates two fragments separated
   by one missing internal beam, as well as bounded scan-endpoint fragments.
   However, its insufficient-witness handler preserved compatible partial history
   only for endpoint fragments. An internal dropout cleared the history before
   three unique scans could accumulate.

   The saved `20260923T144640Z` run, candidate `001_survey_candidate_0001`, contains
   repeated `history_reset` events with reason `not endpoint fragments`. At scan
   stamp `1790175180.9900935`, two witnesses are discarded for raw groups
   `[[0], [2]]`. Two subsequent unique scans at `1790175181.08565` and
   `1790175181.17607` are discarded again at `1790175181.2716043` for the same
   one-beam dropout. The source is the candidate's `camera_lidar_attempt_00/`
   `observer_status.json`, under
   `results/implementation_checks/run_audit_20260923T144640Z/results/aufgabe04/`
   `real/autonomous_exploration/stand_explore_exact2_camera_all5_20260923T144640Z/`.

2. **Head fitting could consume QR evidence.**
   The runtime committed head association to `_scan_target_persistence` before
   decoded-symbol binding previewed that same owner. A rejected off-target head
   could drain or reset the raw witnesses needed by a valid QR ray. A deterministic
   reproduction seeds three real scan records: the QR ray passes before that head
   rejection and fails after it. The broad registration envelope already had a
   separate owner; the narrow QR ray did not.

3. **One registered QR path omitted fragment recovery.**
   The finite-range/parallax fallback used the supplied persistence resolver,
   while direct registered QR association did not. This particularly affected
   calibrated configurations with zero camera-to-scan translation.

The September 29 `135018Z` run has a separate, documented range-hypothesis
failure: 329 QR decodes but no accepted association for candidate 0005. Its bounded
radial recovery was already implemented in revision `6423304`, the starting
revision of this change. The later `144233Z` capture contains nine successful
current-head associations in eleven saved frames and does not establish another
fragmentation failure. These findings must not be presented as a newly reproduced
failure of that latest post-fix run.

## Implemented correction

- Both supported fragment kinds now use the existing partial-proof validation.
  Compatible dropout frames retain earlier witnesses while waiting for three
  distinct, contiguous scans. Fragmented frames never become witnesses.
- QR association has a separate persistence owner. Head, QR and broad-envelope
  owners receive the same exact-time stopped scan inputs, but each evaluates its
  own current bearing. A rejected head cannot change QR history.
- All owners reset on observation/context changes and scan-ingress loss. Runtime
  status includes separate QR witness diagnostics.
- Direct registered QR association can resolve raw ambiguity when no separate
  position-reconciliation proof applies. Already witnessed subsets are not
  submitted again as raw associations. Independent-envelope and identity checks
  still run after resolution.

This changes evidence retention, not the physical association thresholds. It
preserves original scan topology and raw cluster counts, admitted range and
bearing limits, spatial-gap limits, exact-time transforms, stopped-pose checks,
source freshness and neighboring-candidate rejection. The selected cluster
contains only current real returns; historical returns establish continuity but
are never inserted into the current scan.

## Validation and limits

Regression tests reproduce the pre-fix failure of alternating unique scans and
internal dropouts, then verify recovery after the third genuine witness through
both camera-driven and independent scan ingestion. Negative cases cover finite
intervening returns, unsupported fragments, expired support and ambiguous
independent registration. Existing endpoint, proof-tampering and receipt replay
tests remain applicable.

`test_scan_internal_history.py` replays four actual September 23 scans from
`fixtures/scan_internal_history_20260923.json`, with original ranges, exact
recorded TF-derived context and source hashes. The dropout preserves two prior
witnesses but remains unadmitted; the next genuine scan supplies witness three.
This reduced camera-capture sequence omits intermediate independent scan inputs.
It proves retention, not an additional live QR admission.

`test_observer_qr_scan_persistence.py` exercises the runtime's head-then-QR
resolver ordering: a deliberately rejected head consumes its own pending scans,
while QR resolution still proves the current fragments from its independent
history. Separate binding tests round-trip the recovered QR through the full
discovery-receipt validator and verify that reconciliation never consumes
ordinary candidate-history evidence.

Final focused validation: **252 tests and 129 subtests passed** across 21 test
modules, including clustering/topology, both persistence paths, runtime scan
ingestion, QR binding/receipt validation, camera processing, retained opposite
centers and candidate-position recovery. All four recorded fixture source hashes
match the local captures. `git diff --check` passes.

This is a local, offline correction. A permanently fragmented target without
three real continuity witnesses still remains ambiguous. Motion or a changed
candidate/epoch invalidates history; the correction does not carry stationary
evidence through driving. Live admission rate and camera-processing latency must
be measured after deployment.
