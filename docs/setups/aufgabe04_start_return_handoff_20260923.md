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

The separate `return_to_start` leg uses the existing dry-run, certificate,
uncertainty, live obstacle, localization and one-use motion permit checks.
Old authorization scopes cannot silently gain this leg. Existing exploration
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
`return_to_start/failure.json`. The return phase does not send HTTP requests.

Validation: the final 22-module offline regression selection passed **347 tests
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
