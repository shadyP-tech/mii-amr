# Recorded stationary target-support evidence, 2026-09-30

`inputs.json` contains lossless field selections from original capture JSON,
not fabricated sensor messages or precomputed test verdicts. `manifest.json`
records the complete original source-file SHA256 values, the fixture-file hash,
selection rules and independently calculated sequence properties. Original
files remain under the referenced local `results/implementation_checks` audits.
The fixture is self-contained for scan association and evidence-state replay;
those audit directories are not required to run a test.

The three groups are:

- `negative_sequence`: all twenty contiguous saved captures through the first
  replayed terminal receipt, from candidate
  `survey_candidate_0005`, initial camera attempt, stopped run
  `stand_explore_exact2_camera_all5_20260930T150526Z`. Capture 1 exhausted TF;
  captures 2–20 are nineteen distinct processed tuples spanning 5.265656
  seconds in image time. Original observer events bind each processing call's
  exact `last_evidence_source_freshness` value and check time. All nineteen
  are fresh with exact-time TF; none is skipped. The first terminal receipt
  occurs at capture 20, 5.253284 seconds after the first qualifying tuple
  (7.535449 seconds after the observer was invoked). Earlier deadline-limited
  head searches do not invalidate their fresh raw scan evidence.
  The original three-degree nominal cone has no finite return at capture 16.
  The full fifteen-degree registration envelope reports explicit out-of-range
  background in every processed frame, including capture 16. Every frame has
  zero in-range samples; the range gate is roughly 0.368–0.588 m. This broad
  current raw scan replay has no persistence resolver and must not be replaced
  with nominal-cone results or a sparse selection of favorable captures.
- `excluded_tf_retry`: capture 1 from that same attempt. It contains original
  image/scan messages but exhausted exact-time TF retries, has no TF samples
  and no detector result. It must not count as fresh negative target evidence.
  This duplicates the first contiguous capture for a separate exclusion test.
- `valid_control`: capture 1 of candidate `survey_candidate_0003` from run
  `stand_explore_exact2_camera_all5_20260930T135242Z`. Its original preliminary
  association is positive at approximately 0.542 m, selects source scan
  indices `[0,1,2]`, and the current head is detected and target-associated.
  This is an independently recorded positive control, not a mutation of the
  wall sequence.

Each frame retains complete original `sensors` and `tf_samples` subtrees,
including scan endpoint metadata, range strings, camera information and both
map and odometry poses. Receipt, selection and completed-result timestamps,
profile hashes, motion-epoch/target-key observation evidence, original
association results and current-head/freshness flags are preserved. Arrival
target receipts retain their candidate/projection/snapshot hash bindings.
Large detector image-processing diagnostics are intentionally omitted. The
manifest describes the selection, and every retained value can be compared
directly with its named field in the SHA-bound source.
The original event artifact hash and one-based event line additionally bind
each selected source-freshness subtree. TF-only captures have no such subtree.

To replay scans, decode the original `"nan"`, `"+inf"` and `"-inf"` strings
with `float`, construct `PlainLaserScan` with the original header/stamp,
endpoint metadata and recorded topology profile, then call the production
association function with its recorded bearing, range and clustering gates.
For an exact association replay, `now_sec` is the recorded scan receipt plus
recorded `scan_age_sec`. Do not replace the missing TF in the excluded case.
Stationarity can be checked from the `map <- base_footprint` exact-image-time
TF, independently of detector conclusions. Wall-clock sample times must not
be mixed with monotonic receipt times.

No image bytes are needed or included. Original image SHA256 strings are
preserved for provenance. All selected 15:05 source JPEGs were locally
verified against those hashes; the 13:52 control JPEG is absent from that
local audit copy, so its image hash is metadata-only and explicitly marked
`image_bytes_verified_locally: false`. This fixture tests sensor support and
evidence admission, not image decoding or recognition. Tests must not infer
stand identity, delete candidates, or authorize motion from these records.
