# LiDAR → camera handoff fixes — 9 September 2026

The code now handles the reproduced startup, camera publication, scan seam and
identity integration defects. These are offline-verified changes; they have not
been deployed to the robot or validated in a new physical run. The failed
`stand_explore_exact2_camera_20260909T123820Z` run remains evidence of zero camera
observations and zero QR identities.

## Changes and operating defaults

| Area | Implemented behavior | Remaining evidence |
| --- | --- | --- |
| Execution TF listener | A structurally validated stale first global edge may remain stopped during the existing cold-start acquisition deadline. It must become fresh and pass all other gates before motion. Stale history remains ineligible for reseal. The actual execution buffer records bounded receipt/source/latest-buffer timing. | Hardware delivery traces across preflight → execution; no DDS or broadcaster root cause is claimed. |
| Camera publication | Original image **and** scan timestamps are checked again before evidence admission and all three authoritative JSON commits, including after serialization and `fsync`. Expired results cannot set completion flags. Recommendation observation time remains image source time. | Real capture-to-commit timing and repeated jointly valid observations. The 250 ms tracker and 500 ms operational windows have not been enlarged. |
| QR work | One image invocation reuses decoder results only for the same exact crop and decoder mode. Empty/multiple-symbol results and corner coordinates are preserved. Registration/geometry is evaluated again. | Full-frame and tracked-crop latency on the robot. |
| Recording | The mission runner enables bounded asynchronous raw capture per camera attempt. It preserves original compressed bytes, sensor headers and calibration, scan geometry/returns, exact returned TF samples, ROI/decoder/stage diagnostics and quota/drop counters. | Synchronized original images for geometry association and angle-error measurements. Capture files are diagnostic evidence, never pose/motion authority. |
| LiDAR topology | Both discovery and camera association share endpoint-adjacency validation. Only original `N−1 → 0` endpoints may join; original range/distance/cone/size gates remain. Received, dropped and exact-TF timeout/latency counters are recorded. | Original `angle_max` from real scans and measured stand/clutter fixtures. Morphology thresholds were not retuned. |
| Identity/catalog | QR/server identifiers preserve case. Discovery records observations without inventing station IDs. Explicit robot-scoped saved server mappings bind identities. A checked offline adapter revalidates fixed arrivals/corridors and carries the full hypothesis pool into route planning. | Server advancement/acknowledgement semantics and real ordered logistics execution. |

`--scan-topology-profile` defaults to `linear`. Set `full_rotation` only for a
known full-rotation scanner. Even then, **each message** must have finite,
consistent original sample count and angle metadata, and both the reported and
indexed seam gap must be within 0.1 sampling step of one. Two-step gaps, missing
endpoints, partial scans and inconsistent/overlapping geometry stay linear.
The 222/224-return tests use measured increments with explicitly reconstructed
endpoints where the old receipts did not retain `angle_max`; those are not
claimed as original raw scan fixtures.

The outer runner forwards the same topology setting to both observers. Its
per-attempt capture defaults are `--camera-capture-max-frames 64` and
`--camera-capture-max-bytes 33554432`. Queued/in-flight capture is additionally
bounded to four frames and 8 MiB. Original frame bytes plus frame metadata count
toward the run quota; the fixed-size summary is separate. Capture failures and
quota drops are reported without changing perception admission. The passive
observer also exposes `--capture-history-dir`, `--capture-max-frames` and
`--capture-max-bytes` for standalone use.

## Repeatable one-candidate milestone

Add `--stop-after-camera-candidates 1` to an otherwise valid
`execute-exact-two-camera` or `execute-full` invocation. Keep the physical site's
stand count and complete LiDAR candidate snapshot. The option only limits the
camera pilot; existing motion authorization, stopped observation, localization,
collision, QR and facing gates still apply.

After the first validated candidate receipt is committed, the runner saves
`camera_candidate_checkpoint.json` and a mission summary with
`status: camera_checkpoint_complete`, `goal_completed: false`,
`exploration_complete: false` and `motion_authorized: false`. It does not emit a
completed discovery catalog or station registry. A terminal local observation
failure ends the pilot before another stand is selected. Local views/retries
remain bounded by the existing inspection budget. The pilot is not a resume
authorization; a repeat needs its normal fresh session and gates.

Full discovery without server mapping evidence writes
`observed_station_identities.json` and reports `server_binding_pending`; its
station registry path/hash are null. With `--server-qr-mapping-evidence` and
`--server-robot-id`, evidence is checked before candidate motion and again at
binding. If it expires during discovery, the observed identities remain saved
and no bound registry is committed. Saved mappings have content-integrity
seals, not server signatures or scan acknowledgements.

The new catalog bridge requires current, hash-bound recommendation, map,
calibration, robot, source registry and frame-projection evidence. Old facing
catalogs missing these bindings fail promotion explicitly. Promoted catalogs
retain a separately sealed complete obstacle snapshot, which the route consumer
requires whenever its provenance contains that hash.

See [the offline binding and promotion commands](aufgabe04_catalog_promotion.md)
for sealing captured server mappings, promoting evidence and planning with the
complete obstacle pool.

## Validation and scope

Regression tests exercise missing → stale → fresh TF startup under the unchanged
deadline; slow debug output and disk flushes at camera publication; seven actual
observer processing calls establishing consensus followed by expiry; exact-crop
decoder reuse; capture quotas, disk faults and shutdown; seam merges and rejected
missing sectors; exact server identity binding; and pilot stop/completion fields.
The final combined run passed **747 tests and 572 subtests** across 60 modules,
with zero failures, errors or skips, using Python 3.12.14 and the local OpenCV
environment. The source snapshot did not change during the run. The
[validation receipt](../../results/audits/aufgabe04_code_fixes_20260909/validation.json),
[exact command](../../results/audits/aufgabe04_code_fixes_20260909/command.json),
JUnit report, console log and Python source hashes are saved under
`results/audits/aufgabe04_code_fixes_20260909/`.

Independent review reproduced and closed a capture submit/shutdown race and
catalog promotion gaps involving non-QR evidence, old sensor data behind a
newer observation timestamp, and foreign odom frames. Regression fixtures now
reject those cases before publication. The seam, pilot-stop, source preservation
and obstacle-pool changes also received independent review.

The implementation does not supply a real logistics driver, loaded transport or
two-robot execution. Those require the server/runtime and measured physical
contracts described in the readiness audit. Camera border association tuning,
adaptive extra survey viewpoints and geometry thresholds also remain dependent
on real synchronized measurements. The next physical milestone is a repeated
candidate approach → stopped observation → validated QR and facing pose.
