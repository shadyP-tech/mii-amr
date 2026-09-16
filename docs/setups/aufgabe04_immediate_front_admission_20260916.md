# Current head fit and decoded QR admission

Frontside exploration now completes its stopped camera observation as soon as
the candidate has a decoded, associated QR identity and a current accepted 3D
head fit. Both the viewer and observer use `admit_measured_head_model` for that
fit. A displayed proposal or predicted overlay is not an accepted measurement.

`observer/immediate_front_observation.py` joins the independent identity and
geometry channels. It accepts the first bound decode, without waiting for the
legacy two-decode latch or seven simultaneous QR/angle frames. Identity can
precede geometry by at most one second (or the configured QR lifetime if
shorter), within the same target, camera/model context and stationary epoch.
Its recorded QR corners must remain inside the current complete head. This is
an identity-to-head binding; QR size and QR-derived angle do not control the
measured head angle.

Every angle still uses fresh pixels, the measured physical profile, the shared
viewer fit checks, a complete crop and current candidate/LiDAR association.
Motion-epoch reset frames, conflicting identities, unassociated scans and
expired sources cannot commit. Image/scan freshness and QR lifetime are checked
again before the durable artifact is published.

The recommendation uses schema 3 and policy `current_head_and_bound_qr`.
`axis.sample_count` is **1**; `axis.confidence` is **0**, explicitly indicating
that no temporal-consensus confidence is claimed. The receipt retains the head
quality and independently bound QR evidence. Planner and arrival-catalog
consumers validate the full receipt before accepting the one-sample policy.
Route clearance, localization, final facing-pose validation and motion
authorization continue independently. Backside classification and the bounded
orientation fallback retain their existing observation requirements.

Acquisition uses the viewer's cold head locator first inside a bounded region
around the projected candidate. It reserves processing time for the associated
current fit, then uses the existing projected search if appropriate. A verified
head locator survives up to two fresh, same-context misses within its original
two-second lifetime; misses never refresh its evidence or supply an angle.
Acquisition diagnostics retain the stage that exhausted a deadline.

The existing experiment command still applies, including
`--axis-sample-count 7`; that flag controls the bounded/backside and legacy
consensus paths. No extra flag enables immediate front admission.

Validation covers first-frame admission through the real observer processing
and status paths, QR-before-geometry, identity expiration during serialization,
motion/conflict/crop/scan rejection, honest planner/catalog sample counts, and
bounded acquisition/tracking. Synthetic and offline tests do not establish
hardware detection rates. No run files were transferred and no robot motion
was executed for this change. In particular, independently ambiguous QR/LiDAR
clusters still reject; this change does not repair inconsistent scan metadata.

The final focused observer, acquisition, artifact, planner, catalog and legacy
regression run passed **375 tests and 472 subtests**. `git diff --check` passed.
