# Bounded opposite-side localization checkpoint

The `20260928T150641Z` run classified the backside correctly but rejected seven
opposite routes before motion because their uncertainty budgets exhausted the
available clearance. A stationary refresh alone did not admit the routes.

The opposite coordinator now tries one stopped checkpoint after its ordinary
standoff search and one stopped localization refresh have both exhausted their
verified initial-child, no-motion, no-permit uncertainty alternatives. Static
infeasibility, missing validated target center, missing localization readiness,
and failures after a motion permit do not qualify.

## Execution contract

1. Evaluate at most 32 existing interior waypoints per rejected route from the
   final planning epoch. The prefix keeps the measured heading anchor and all
   uncertainty reserves. Require at least 20 cm start-to-stop displacement and
   15 cm remaining route length.
2. Evaluate the hypothetical suffix with the same covariance at the proposed
   stop. This forecast ranks feasible splits by their smaller clearance margin;
   it cannot authorize motion or establish a new live anchor. Both endpoint
   turn envelopes and the ordinary route budget are checked.
3. Seal an exact prefix of the validated opposite route, retaining a hashed
   reference to its full-route diagnostics, candidate snapshot and backside
   geometry. Its terminal yaw follows the incoming segment. This stop is
   explicitly not camera arrival. Parent geometry, clearance, prefix vertices,
   endpoint and evidence bindings are revalidated by the child.
4. Run the prefix through the existing dry/live checks and a separate single-use
   opposite-face permit. The child recomputes the stopped endpoint envelope from
   its fresh covariance. Startup/runtime reseals are disabled for this prefix
   so a checkpoint cannot silently become a full stand approach.
5. At the stop, admit fresh stationary localization, reproject the original
   backside receipt (including angle uncertainty and validated center), and
   replan the opposite approach. Use distinct artifacts, child IDs and permits.
   Do not reuse the forecast covariance or refit a camera angle.
6. A failure after checkpoint dispatch is terminal for this attempt, including
   a rejected suffix or failed fresh localization. It cannot trigger another
   checkpoint, another no-motion refresh, or a stale ordinary local-view move.

The new mission `RUN` scope explicitly includes this bounded checkpoint. Old
mission authorization receipts retain their former behavior and do not grant
checkpoint authority. No command-line option changes are required for a new
mission using the updated checkout.

## Modules

- `navigation/approach/opposite_checkpoint_selection.py`: pure budget selection.
- `navigation/approach/opposite_checkpoint_route.py`: exact-prefix artifacts and
  parent binding validation.
- `real_robot/candidate/opposite_checkpoint.py`: bounded execution handoff.
- Existing opposite orchestration, route validator, uncertainty admission and
  permit checks connect those modules to the real runner.

## Validation

The compact fixture `tests/aufgabe04/fixtures/opposite_checkpoint_20260928.json`
contains derived geometry and covariance from all seven recorded rejections.
All original margins reproduce within 1e-9 m with unchanged limits. On the
best recorded route, waypoint (-1.545, -0.265) m retains approximately 4.60 cm
prefix and 4.77 cm hypothetical suffix clearance, including the conservative
stop/initial turn envelopes. This is an offline feasibility result; the future
live pose and covariance must still pass admission.

Tests cover parent/prefix/candidate tampering, separate sealing, original
receipt retention, checkpoint count and stage ordering, terminal failures,
legacy permit scope, and fresh child covariance rejecting an optimistic preview.
The related regression suite passed 159 tests. After adding the final missing-
arrival guard, 29 handoff/coordinator/permit tests passed, including the two new
negative cases. `git diff --check` passed.
No robot run or workstation deployment was performed for this implementation.
