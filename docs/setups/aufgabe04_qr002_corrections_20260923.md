# QR_002 association and retained-backside completion corrections

Implemented locally following `aufgabe04_qr002_front_facing_audit_20260923.md`.
No robot execution, workstation deployment, ROS nodes or motion commands were
performed by this implementation task. Existing unrelated return-to-Start
work in the shared checkout was preserved.

## Behavior

- QR search and center reconciliation can consume the same recomputable
  stopped-scan fragmentation proof. Raw cluster counts, beam indices, range
  limits and scan topology remain intact. Broad and narrow searches maintain
  separate histories, sharing exact-time scan ingestion rather than TF lookups.
- Historical raw scans are rechecked against each current ray and cone. A
  historical narrower head cone cannot exclude competitors for a wider query.
  One missing internal beam is also supported when a real, validated circular
  cluster crosses index zero; invalid endpoint metadata is not made circular.
- If broad registration remains ambiguous, `qr_finite_range.py` evaluates the
  decoded symbol's calibrated ray over the entire admitted range interval.
  Its full angular envelope must contain one current provable target. That
  target then supplies measured range for the final narrow ray. No head angle
  or head corners are used on this path.
- `qr_ray_candidate.py` separately checks the original candidate bearing/range,
  compactness and displacement, and excludes every other frozen candidate.
  The candidate snapshot and range proof are persisted and revalidated in the
  QR receipt. A competing return inside the QR ray or a competing candidate
  still rejects admission.
- Once a backside angle is ready, a maximum 1.5-second opportunity lets the
  independent center check finish. Both strict and bounded backside commit
  paths use it. Same-epoch misses cannot renew the timer. Centering/inspection
  advice does not preempt this opportunity; timeout remains bounded recovery.
  The existing backside geometry window continues until the receipt commits.
- Opposite-side QR completion retains the certified backside orientation and
  uncertainty, without another front-side angle fit or head-corner prerequisite.
  `retained_facing_center.py` uses the projected validated source center, or a
  separately validated three-scan arrival center if an older angle receipt
  lacks one. The latter evidence is bound to the current QR, candidate, pose,
  epoch and original orientation chain.
- The selected front outward normal is the opposite of the observed backside
  normal. The undirected axis and its angular uncertainty are preserved. The
  existing route and continuous-clearance checks still run before facing is
  marked ready. Discovery remains valid when facing validation fails.

There are no new launch flags. Ordinary QR geometry grace remains configured
by the existing option; opposite-side associated QR identity still completes
immediately.

## Evidence and limits

The checked-in `qr002_cluster_20260923.json` fixture contains 112 recorded
current scan/TF contexts and 67 distinct historical scan witnesses extracted
from the original run's persisted proofs. It is a partial scan history, not a
complete ROS bag. Each source compressed image was downloaded read-only and
checked against its recorded SHA-256.

QR corners were decoded offline with the production decoder wrapper and local
OpenCV 5.0.0: 107/112 frames decoded. These are explicitly offline results, not
the workstation's original OpenCV 4.5.4 corners. Running the same offline
inputs with and without the new ray-resolution path gives:

| QR association | Accepted frame indices |
| --- | --- |
| Previous broad-envelope prerequisite | 35, 41, 112 |
| New candidate-checked QR ray | 32, 35, 39, 41, 57, 76, 100, 112 |

The remaining frames are not declared admitted. In particular, insufficient
independent scan witnesses remain a real limitation. This result does not
establish a new real-robot completion time or precise front-facing angle.
QR_002 has no historical backside angle in this run, so uncertain geometry may
still yield QR-only completion.

Focused verification: **202 tests and 81 subtests passed; one test skipped**
across the association, scan-persistence, observer, bounded geometry, opposite
identity, retained-facing and catalog-projection suites. New regressions cover
the seven-angle/one-center readiness mismatch, bounded expiration, no-refit
arrival recovery, full receipt round trips, internal/seam fragmentation,
conflicting symbols, stale sources, modified evidence and competing candidates.

The broader module-inventory checks have two existing failures (outdated exact
file lists). Both were reproduced against HEAD's test source and HEAD's file
inventory, independently of this change. Those inventory tests were not
rewritten to hide the failures.

A conservative arrival scan-surface center can have larger uncertainty than
a metric head center. The regression explicitly preserves its 0.11 m bound:
a feasible 0.60 m facing endpoint is accepted in the synthetic fixture, while
the 0.35 m endpoint is rejected for excessive worst-case obliquity. Neither
angle reuse nor successful QR decoding silently reduces that uncertainty.
