# Shared physical-frame selection correction — 2026-09-16

This implements the border-selection correction from the
[first-candidate audit](aufgabe04_first_candidate_viewer_parity_audit_20260916T133025Z.md).
It fixes selection/binding order; it does **not** establish reliable camera
admission at the first inspection point on hardware.

## Implementation

The physical path now performs:

1. Locate bounded rectangle hints from the current image.
2. Refine each hint with the same enclosing-physical-border policy used by the
   metric fitter, before comparing or selecting physical frames.
3. Deduplicate current supporting-rail equivalents. Different nested frames,
   conflicting resolutions and nontransitive rail-family bridges remain competitors.
4. Recheck the selected head's candidate/LiDAR association and recenter its crop.
5. Solve the 3D pose on exactly that current measured boundary, retaining the
   selected-border binding check and existing geometric quality gates.

Early association previews still prune candidate-ineligible search hints; they
are not admission receipts. The selected physical head receives the final fresh
association check before the 3D solve.

Modules:

- `perception/stand_axis/current_head_refinement.py`: shared raw-border
  refinement, including cooperative deadlines between OpenCV calls.
- `perception/stand_axis/head_frame_resolution.py`: bounded, image-local
  grouping of measured frames and tested hint mappings. Containment alone is
  insufficient to merge frames. Untested independent hypotheses still reject
  incomplete comparison at the existing budget.
- `perception/stand_axis/current_head_refinement_proof.py`: transient evidence
  bound to the actual source image, model, selected corners and raw head pixels.
  Recentring requires an exact view of the same image and translates the evidence
  once. Copied, mutated or mismatched inputs reject instead of launching another
  search. This evidence contains no historical angle or navigation authority.
- `real_robot/observer/camera_context.py`: normalizes finite ROS/NumPy CameraInfo
  values to native Python numbers before tracking/admission context creation.
  Invalid calibration resets evidence. Strict artifact validation is preserved.

The evidence carrier passes through cold acquisition, candidate registration,
the metric pipeline and the fitter. It is not serialized in head proposals or
retained by trackers. Geometry/QR decoration reuses the same current fit.

Unique current-scan proposal previews also avoid repeatedly registering pending
historical scans. Current scan inputs, association and freshness are still checked;
ambiguous/fragmented cases retain the existing historical proof. Final resolution
still updates the real history. Per-image static context is prepared once.

No CLI flags, route policy, confidence thresholds, physical dimensions or production
full-image-search policy were changed. Existing neutral-hint outward-search semantics
were retained after an experimental anchor change altered a recorded backside rail.

## Validation and practical limits

The focused tests exercise actual cold acquisition → unique LiDAR association →
recentered crop → one 3D solve; QR decoration does not repeat that solve. Additional
tests cover different nested frames, ambiguous family bridges, invalid/mutated
evidence, expired acquisition budgets, malformed calibration and NumPy-backed ROS
calibration through the QR fallback path.

The final complete `tests/aufgabe04` run reports **3,697 passed, 21 failed,
1 skipped, and 3,182 passing subtests**. A clean HEAD archive under the same local
Python environment reports **3,660 passed, 22 failed, 1 skipped**. Every remaining
failing case also fails in that baseline; no new failing cases remain. The suite
is therefore not globally green. Comparison details are recorded in
`shared_frame_test_summary.json`. `git diff --check` passes.

Some recorded expectations deliberately changed with the corrected comparison:

- The tight `backside_000022` crop selects its physical frame before solving and
  passes the unchanged three-degree yaw-uncertainty gate. Tests bind the fitted
  corners to the selected corners; this fixture is not angle ground truth.
- The wider `back22` crop exposes competing current left rails around x=41–42 and
  x=45. It now rejects at physical-frame comparison instead of preserving the old
  early-selected angle. This availability limitation is explicit; thresholds were
  not relaxed to recreate the former overlay.
- The existing explicit-seed `backside_000012` fitting behavior is preserved.

Read-only, in-memory patched replay on mii001 used the original 53 viewer images
and 55 processed mission captures. No images or recordings were copied, no remote
checkout was edited, and no ROS node or robot command ran. Source hashes were
unchanged. Replay forces loss of the causal pose hint at sequence indices 0, 18
and 36; all subsequent hints originate only in earlier accepted replay frames.

| Saved-pixel causal replay | Usable head fits |
| --- | ---: |
| Viewer frames within the recorded mission candidate crop | 0/53 |
| Mission frames within their recorded candidate crops | 18/55 |
| Viewer full image, diagnostic only | 52/53 |
| Mission full image, diagnostic only | 55/55 |

In the bounded mission replay, capture 47 first cold-acquires and 17 later frames
retain a usable current fit. Cold proposal-budget exhaustion remains the dominant
failure. The full-image results are diagnostic, not a production fallback or a
proof of correct physical border/angle ground truth. They do not justify removing
candidate association. This correction alone therefore does not demonstrate
reliable cold acquisition at the first inspection point.

The replay isolates geometry: it does not emulate live camera/scan freshness,
ROS execution, QR decoding, tracking TTLs or the complete observer's changing crop.
It must not be reported as 18 real candidate admissions or a successful robot run.

Saved-scan microprofiling of 16 proposals reduced association plus preview from
approximately 39–61 ms with pending scan history to about 7 ms. This uses preserved
logical timestamps and reconstructed history; it is not a live latency guarantee.

Numerical evidence and replay sources are under
`results/aufgabe04/debug_audits/stand_explore_exact2_camera_all5_20260916T133025Z/`:
`shared_frame_cold_tracking.json`, `replay_cold_tracking.py`,
`shared_frame_scan_microprofile.json`, and `shared_frame_patch_manifest.json`.
Only numerical diagnostics returned from mii001. Production changes remain local
and require their ordinary deployment/hardware validation before being treated
as demonstrated robot behavior.
