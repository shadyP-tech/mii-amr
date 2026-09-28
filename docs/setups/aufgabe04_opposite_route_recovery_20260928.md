# Opposite-face route recovery correction

Addresses the failure audited in `aufgabe04_run_audit_20260928T121706Z.md`.

## Behavior

Generic inspection and certified opposite-face planning now share `bounded_inspection_standoffs`. Opposite planning supplies the uncertainty of the validated retained center. The lower bound includes the active standoff plus center uncertainty and the transit radius plus center uncertainty plus the map half-cell diagonal. The planner still checks actual rasterized keepouts, every candidate, continuous clearance, and route uncertainty; the proposal floor is not a clearance certificate.

For the recorded 0.50 m request, 0.34 m transit radius, 0.05 m map, and 0.025120 m center uncertainty, proposals are **0.50, 0.45, 0.400477 m**. The invalid 0.35/0.33 m requests are never generated. A center bound that leaves no available radius returns a typed candidate-local route failure. Malformed numeric inputs and corrupt evidence remain fatal.

If all proposals fail and at least one dry child produced a verified route-uncertainty rejection **before motion and before any permit**, the new `opposite_localization_retry.py` policy permits **one** additional stationary planning epoch. Production uses the existing stationary-frame admission effect, then reprojects the original backside receipt and candidate snapshot into that fresh frame. Route paths, child IDs, and permit paths are distinct for the second epoch. Every candidate route still passes the normal dry and execute gates; uncertainty thresholds and keepouts are unchanged.

The refresh succeeds only if the newly planned route passes fresh admission. An unchanged uncertainty envelope remains rejected. After the single opportunity is exhausted, `CandidateInspectionRouteUnavailableError` returns control to bounded local-view recovery, which saves both epochs' rejection evidence. Rejected routes do not count as camera views. Static-only exhaustion does not spend a localization refresh. No retry is allowed after a child reports motion, a permit, a wrong fault, or accepted uncertainty evidence.

The implementation uses stationary convergence as the clearance-recovery option. It does not introduce a staged motion protocol. No new CLI flags are required.

## Modules

- `real_robot/candidate/inspection_route_search.py`: shared, finite standoff proposals including optional retained-center uncertainty.
- `real_robot/candidate/opposite_localization_retry.py`: bounded stationary retry policy and epoch evidence.
- `real_robot/candidate/approach.py`: fresh frame admission, original-evidence reprojection, and certified route orchestration for each epoch.
- `real_robot/candidate/inspection_execution.py`: persist structured exhaustion evidence before continuing local-view recovery.

## Verification

Focused validation: **98 tests passed, 38 subtests passed** across opposite localization retry, autonomous candidate approach, inspection route search/execution, opposite-face fallback, backside frame projection, and retained-facing suites. Run with Python 3.14 / OpenCV 5.0.0 in a temporary local environment. `git diff --check` also passed.

`tests/aufgabe04/test_opposite_localization_retry.py` covers the recorded failure sequence, distinct epoch identities, retained receipt/normal reuse, no retries after motion/permits, malformed failures, fixed retry count, and success after a refreshed admission. Its derived fixture `fixtures/opposite_route_20260928.json` contains recorded geometry and uncertainty inputs with source run/revision/map hashes; it is explicitly not an admission receipt.

The real-map geometry regression reproduces the 0.40 m blocked stand-keepout cell and the 0.45 m route's **−0.021682 m** budget margin. A separate synthetic improved covariance verifies that admission can become positive under unchanged physical limits; this is not a prediction of live AMCL convergence.

Controller regression verifies that exhausted opposite routing continues to a local view and immediate QR-only completion while retaining the structured failure record. Existing candidate-orchestration fixtures now provide a real map resolution when testing decreasing offsets instead of assuming a physical-only minimum.

The broader autonomous-runner suite has one baseline failure in `test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`: its synthetic odom certificate lacks `odom_execution_certificate_sha256`. The same failure was reproduced with all changed candidate modules loaded from HEAD; it was not caused by this correction. The other 63 runner tests passed with local Unix-socket support enabled.

No workstation deployment or real robot run was performed. Live admission can still reject an unsafe opposite route; the correction provides bounded recovery rather than guaranteeing admission.
