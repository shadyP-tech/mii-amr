# Candidate-guided head acquisition — 2026-09-17

Camera exploration now uses the selected candidate and current scan while
comparing head proposals, and has a bounded fallback for heads whose observed
rails are displaced by window/radiator segments in the initial line quotas.
This applies to the ordinary exploration observer, including inspection after
the opposite-side branch. Existing arrival, stationary, freshness, tracking
reset, QR binding and final association requirements remain in force.

## Automatic location hints

The manual blue/cyan/magenta diagnostic supplies a hand-chosen quadrilateral and
a 0.93 scale before fitting current pixels. Its good returned fit demonstrates
the value of a useful initial location; it does not establish automatic location
or distinguish every physical-border interpretation. The new locator has no
manual coordinates, fixed 0.93 shrink, decoded QR rectangle or prior pose.

The original full-image search retains its first decision. A completed miss can
try the existing size-priority distribution. If that also completes without an
accepted head or unresolved alternative, a final pass compares observed rail
pairs **before** allocating its 24-line-per-direction quota. Candidate center
and size bounds exclude incompatible pairs; current four-side support and
corner arms earn compatible rails priority. Spatial buckets keep one cluster
of background segments from taking all available slots. Short perspective or
neck-split rails can qualify through the complete observed pair.

That final pass also probes small normal offsets around up to 32 observed
quadrilateral hints. It keeps at most eight coherent, current-edge-supported
variants per hint within the existing model refinement corridor. This provides
an automatic counterpart to adjusting the diagnostic's initial location. All
variants remain unverified hints until the ordinary raw border, enclosing-frame
and 3D quality checks succeed. The original hints and raw image remain intact.

All three attempts share the original deadline and **12 strict verification
slots in total**. Success, ambiguity, unverified alternatives, exhausted budget
and deadline expiry stop the search. A rejected background association is a
completed miss and may use remaining slots. There is no fourth attempt. The
guided source has explicit bounds of 512 eligible lines per direction and 256
supported pair hints; overflow rejects instead of claiming uniqueness. Each
corner check also observes the shared deadline.

## Earlier current-scan filtering

Rough hints use only a conservative camera/map bearing envelope, allowing their
center to move up to ten pixels during refinement. They do not traverse scan
clusters. Completed measured proposals are checked against the current scan's
accepted range, registered cone, cluster evidence and age before competing for
selection. This prevents a geometrically plausible window outside the stand's
scan association from blocking an otherwise usable head.

Persistence previews are read-only. The ROS clock is checked for each measured
proposal, and the observer still performs its final ordinary scan association
and freshness checks. A scan-dependent callback bypasses the image geometry
cache so changing scan/time context is evaluated again; exact-image edge
preprocessing may be reused. Debug images without recorded candidate or scan
context keep the unconditioned path.

## Verification and limits

The adversarial current-pixel regression contains 30 horizontal and 30 vertical
radiator-like segments which displace the actual head's rails from both earlier
24-line distributions. The final pass recovers the real four-border head in one
strict fit, retaining proof bound to the original image. Separate tests cover
short sides, missing/clipped borders, inward/outward alternatives, earlier
association rejection, deadline expiry, shared budgets and unchanged ambiguity.

The broader regression run passes 325 tests and 260 subtests. The final three
new test modules pass 28 tests and 17 subtests after adding the association-retry
and per-corner deadline regressions. A focused independent review found no
remaining material defect. No robot deployment or motion was performed.

The final replay preserves all 60 usable geometry-only results across 150
recorded images, including unchanged geometry and ordered strict checks. With
actual recorded scans, 53 of the 80 mission images yield usable geometry:

| Recording | Geometry-only baseline | With early recorded-scan filtering |
| --- | ---: | ---: |
| Opposite inspection `093041` | 5/15 | 6/15 |
| First inspection `133025` | 54/55 | 46/55 |
| Off-center inspection `150723` | 1/10 | 1/10 |

Opposite frame 7 gains a foreground head because two window alternatives fail
the recorded accepted-range check. The eight removed first-inspection results
have ambiguous registered scan clusters; their geometry alone did not establish
admission, and isolated replay supplies no invented persistence history.
Two of those rejected images activate the new final pass and remain unavailable
after 12 total checks. No replay exceeds the shared strict budget. Thus the real
recordings demonstrate preserved geometry and earlier scan discrimination;
successful final-pass recovery is established by the adversarial regression,
not a new acquisition success on these recordings.

Matched timing uses the same first/middle image from each recording, three
unprofiled calls per image. Most medians remain approximately unchanged. For
`133025`, extra attempts on the scan-rejected middle image increase the sample
median from 54.50 ms to 95.15 ms and p95 from 60.94 ms to 142.16 ms. This change
does not establish an overall speedup. The original deadline remains binding
in live use; the replay isolates geometry/scan compatibility with a recorded
clock and does not certify live source freshness. All 11 replay overlay module
hashes match the final working tree.

Recorded-image results and matched workstation timing are reported in the
[replay audit](../../results/aufgabe04/implementation_checks/guided_head_hints_20260917/README.md).
The committed comparison baseline is `ebd73d8`; the earlier
[candidate-screen report](aufgabe04_candidate_head_screen_20260917.md) compares
against `35aa6aa` and describes the preceding implementation stage.

Nearby left-edge alternatives remain unresolved when they both satisfy current
border and 3D quality checks. Scan association cannot distinguish them if they
share the same registered stand evidence. Trials using calibrated upright
orientation and the independently observed printed-panel inset did not provide
a validated discriminator on these images. The implementation therefore does
not claim to reproduce the manual diagnostic's 49/50 success rate automatically.
