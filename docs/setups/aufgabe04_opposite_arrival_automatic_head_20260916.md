# Automatic current-head acquisition after opposite-side arrival

The camera exploration observer now carries a fully measured current head border from automatic acquisition into the same 3D fitter used by the camera diagnostic. This builds on the shared-refinement and camera-context changes already in the working tree.

## Implementation

- Apply the observer's existing vertical candidate bound before allocating the cold-search comparison budget. Candidate height prioritizes complete rail hints over printed texture fragments; position and scale never supply accepted corners.
- Keep small supported paired-rail texture hypotheses for the existing composite-rectangle rejection policy.
- Resolve an already measured initial border through another completed raw-border recovery only when its original rails agree and the recovered proposal has independently passed candidate association. Carry the existing measured final proposal; never construct corners by combining borders. Conflicting recoveries, unrelated enclosures, untested hints, and nontransitive chains retain their ambiguity.
- Preserve the chosen current measurement across crop recentering. Its proof checks image ownership, exact view, image digest, model and selected corners. The fitter consumes the captured raw edges because Canny hysteresis can otherwise remove interior border evidence when the same pixels are cropped. It still performs the ordinary 3D solve and quality gates.
- The opposite-side route keeps the same candidate identity. The existing immediate-front admission accepts one fresh complete associated head plus an independently bound QR; the new regression exercises receipt publication and reload after the certified-backside transition.

There are no changes to the source-age limit, QR identity requirements, candidate association, physical-model quality gates or robot motion policy.

## Validation

Local targeted suite: **280 tests and 318 subtests passed**, using Python 3.14.7, OpenCV 4.13 and NumPy 2.5.3. Additional affected seed, QR pose and scan geometry tests: **31 tests and 53 subtests passed**. This includes real-Canny crop hysteresis, cluttered rail selection, provenance conflicts, stale/predicted geometry, missing/neighbor QR and candidate receipt mismatch.

Offline replay uses the workstation's Python 3.10/OpenCV 4.5.4 runtime and the original opposite-side arrival recording `stand_explore_exact2_camera_all5_20260916T093041Z`. Local changes are supplied through an in-memory importer; the workstation repository and recordings remain read-only. Source and overlay SHA-256 hashes are saved in the replay output.

Of the 15 saved captures containing candidate-projection metadata, **7 produce an automatic accepted metric head fit**, with no manual location or angle seed. Eight remain ambiguous and are rejected. Median acquisition-plus-fit time over these 15 captures is **94.1 ms**. This geometric isolation does not by itself establish candidate admission: it intentionally omits scan association, QR binding and the live deadline.

The second replay uses the real recorded scan, transforms, calibration, candidate dimensions/range bounds, QR decoder, scan-persistence previews, cold-search schedule, current-head association and immediate-front gate. It retains each frame's recorded processing-start sensor age and advances it by measured processing time, including image preparation. Every frame starts with empty tracking, QR identity and scan-witness state; the calibration-only rectification cache matches the observer.

Under the unchanged **500 ms** source-age limit, **4/15** frames obtain a usable candidate-associated head fit. **Frame 44 reaches immediate-front admission readiness** with independently decoded, bound QR identity `Start`, at **422 ms image age** and **131 ms processing time**. The other frames yield six head-processing deadline rejections, five unresolved-border rejections, and three usable fits without bound QR identity. These counts describe this offline replay, not an estimated live success rate. Timing depends on runtime load and initialization; actual ROS scheduling and publication were not replayed. The separate transition regression covers receipt publication and parent candidate continuity.

Reproduction scripts and machine-readable evidence are in `results/aufgabe04/implementation_checks/opposite_side_front_20260916/`: `build_overlay.py`, `replay_arrival.py`, `replay_registered_arrival.py`, `measured_recovery_full.json`, `registered_replay.json`, and `targeted_tests.txt`.

These are offline results. No physical run or deployment has been performed, and replay does not establish a physical orientation ground truth or a live success rate.
