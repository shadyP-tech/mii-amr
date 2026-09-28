# Stored Start pose handoff

Full and exact-two camera missions now store their completed discovery artifacts,
then return to the admitted robot pose associated with exact QR text `Start`.
Camera pilot and coverage checkpoints stop at their existing boundaries.

The handoff authenticates the saved goal, identities, candidate snapshot,
catalogs and observation-frame provenance. Geometry records supply their facing
pose; QR-only records supply the stopped observation pose. Both retain yaw.
The complete candidate obstacle pool and target are reprojected through odom
into a freshly admitted map frame before route planning. The route preserves
the exact endpoint, checks continuous candidate clearance, and smooths safe
segments. A stationary heading correction uses a dedicated isotropic point
clearance budget and commands no translation.

The return now checks uncertainty before motion, using the same stopped
covariance envelope, robot geometry and 0.075 m braking reserve as the faster
child. It takes the full route when admitted. Otherwise it selects a certified
prefix, preferring existing route vertices and then bounded cuts on long
segments. The selected prefix keeps the original heading anchor; it cannot
numerically reset uncertainty before physically stopping.

At most four `return_to_start` legs are allowed. Each uses the existing dry-run,
certificate, uncertainty, live obstacle, localization and one-use motion permit
checks. Initial and intermediate endpoint turns retain worst-axis point
clearance checks in both preview and fresh child admission. The stored Start
target and yaw remain separate from intermediate endpoints, with the complete
planned route and exact selected prefix bound by hashes and geometry checks.

After each completed child, the runner obtains a fresh stationary frame,
reprojects the stage target through odom, verifies arrival and measured progress,
then reprojects the original Start pose and full candidate pool for the next leg.
That stopped arrival epoch also supplies the next planning covariance. Failed
children or arrival checks stop the sequence. The new mission scope permits up
to four legs; the legacy single-return scope can authorize only final stage 0.
Exclusive per-master stage claims prevent duplicate indices, skipped stages,
target substitution and continuation after the final leg. Existing exploration
recovery budgets do not extend to the return.

| Control phase | Linear cap | Angular cap |
| --- | ---: | ---: |
| Return cruise | 0.15 m/s | 0.60 rad/s |
| Return final approach / corners | 0.055 m/s | 0.18 rad/s |
| Stationary final alignment | 0 m/s | 0.18 rad/s |

The fast policy requires physical unloaded odom execution, sensor ages at most
0.25 s, a 0.075 m braking/latency reserve and unchanged obstacle stop/slow bands.
Its command envelope is saved in the dry preflight and hash-bound to the live
permit. These are configured limits, not hardware-validated maximum safe speeds.

`mission_summary.json` records camera completion before motion and keeps
`fastapi_request_ready=false` until successful child completion and a fresh
stationary arrival check. The arrival tolerance is 0.08 m and 0.15 rad; ordinary
child terminal control remains tighter. Successful evidence is published at
`return_to_start/arrival.json`; failures retain discovery artifacts and publish
`return_to_start/failure.json`. Per-leg planning, permits and arrival evidence
live under `return_to_start/legs/000`, `001`, etc.; intermediate arrival records
keep server readiness false. The return phase does not send HTTP requests.

The correction follows the [14:46 run audit](aufgabe04_start_return_audit_20260923T144640Z.md).
Its compact regression fixture includes the actual map occupancy, all six
candidate geometries, original route and covariance. The original full-route
margin reproduces at −0.223007 m. Tests run the actual planner and admission
through the staged mission with simulated stopped poses; the recorded case
reaches the exact stored Start pose in two legs. A fresh covariance deterioration
at the first stop prevents the second motion. These are offline tests, not
physical validation.

Staged correction validation: the final 19-module offline selection passed
**293 tests and 284 subtests**. One pre-existing legacy recovery fixture was
excluded for the missing certificate hash described below. The selection covers
the recorded planner-to-mission pipeline, changed covariance, stage arrival and
frame reprojection, prefix/source tampering, independent child turn envelopes,
bounded sequential permit consumption, old-scope compatibility, speed policy
and the autonomous mission wrapper. Compilation and `git diff --check` passed.
No workstation deployment, robot motion or HTTP request was performed.

Initial handoff validation (before the staged correction): the 22-module offline regression selection passed **347 tests
and 348 subtests**, with compilation and `git diff --check` passing. It covers
both pose kinds, hash/identity tampering, exact-two retained registry history,
full-pool keepouts, exact endpoints, frame changes, false child completion,
failed arrival publication, dry/live speed binding, stationary point budgets,
odom certificate persistence, zero translation under drift and normal mission
authorization/recovery behavior. A 10 Hz closed-loop corner-route test completed
over 20% faster with the new cruise caps while preserving precise corner/goal
control.

One known legacy test was excluded from that combined selection:
`test_runtime_localization_recovery_uses_scoped_permit_without_parent_input`.
Its fake odom certificate lacks `odom_execution_certificate_sha256`; the same
failure was reproduced with the unmodified HEAD recovery-authorization module.
An earlier broader run also encountered existing explicit module-inventory
expectations that omit many pre-existing camera modules. No robot motion or
FastAPI requests were performed during validation.
