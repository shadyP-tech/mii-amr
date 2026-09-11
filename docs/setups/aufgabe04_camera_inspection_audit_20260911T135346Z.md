# Camera inspection audit: first candidate and additional views

The latest run, `stand_explore_exact2_camera_all5_20260911T135346Z`, **uses the independent measured 3D head model for accepted head angles**. The first camera point decoded `QR_003` and classified `survey_candidate_0003` as `front_readable`. Additional views were requested because the observer never accumulated the required seven fresh, associated head-angle samples together. The controller requires a joint QR/facing recommendation to finish the candidate; the recorded classification was advisory progress.

The run used clean commit `75d625f2c97ceb3258a215de2110f5853d007a5e`. This audit copied **619 files** at **2026-09-11 14:07:51 UTC / 16:07:51 CEST** and verified their SHA-256 hashes. The bundle records hostname `mii0002`; retrieval used alias `mii001`. At the snapshot and subsequent read-only status check, the parent and `inspection_002` manifests had no end time or exit code, and no mission failure artifact existed. This is an audit of the observed decision sequence, not a claim that the complete mission terminated.

## Which angle is used?

The autonomous observer invokes `estimate_stand_axis_from_metric_model` with crop-adjusted intrinsics ([node.py](../../scripts/aufgabe04/real_robot/observer/node.py), line 1309). For the measured physical profile, current head borders, corners and neck evidence feed the independent head fit. [head_model_fit.py](../../scripts/aufgabe04/perception/stand_axis/head_model_fit.py), lines 74 and 131, fits the measured head plane and takes the accepted pose's yaw. The live source is `model_current_measured_head`.

The purple dashed rendering visualizes model geometry. Its head depth and stem are projected geometry; a diagnostic overlay can also show a prediction or proposal. Its appearance is not itself evidence of an accepted current fit. [stand_model_overlay.py](../../scripts/aufgabe04/perception/debug/stand_model_overlay.py), line 67, identifies those projections as diagnostics. The autonomous observer accepts current measured head pose only after quality, freshness, candidate association and temporal evidence gates. Its head angle does not depend on the legacy 35-degree limit, and this run contains no rejection at that limit.

## First camera point: detection succeeded, joint admission did not

The candidate approach completed successfully, with its bundle ending at **13:58:19 UTC**. The first observer recorded statuses from **13:58:25.380 to 13:59:12.541 UTC**. The stand, head and QR are clearly visible in the saved full frame:

![First camera point, original saved frame](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T135346Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_00/perception_debug/latest_frame.png)

Recorded outcomes across **186 fresh processed detector results**:

| Result | Count | Consequence |
| --- | ---: | --- |
| `head_neck_junction_gap_too_large` | 146 | Head angle unavailable; includes the final result committed as advisory progress. |
| `head_model_planar_axis_ambiguous` | 1 | Head angle rejected. |
| Usable measured head fit | 39 | Estimated yaw **31.14–33.66 degrees**. |
| Of those 39, rejected by LiDAR association | 24 | All report `camera_bearing_outside_map_cone`. |
| Remaining admitted angle frames | 15 | Peak **4/7** retained within the **5-second axis TTL**; two remained at exit. |

The observer reports 144 QR-sample frames and a latched identity `QR_003`. There was no motion-epoch reset or poisoned identity. A nonrenewable **30-second stopped front-recovery budget** began at 13:58:28.608; its eventual advisory records `front_recovery_budget_exhausted`, 132 qualified recovery frames, `front_readable`, and no camera-relative yaw recommendation. These facts establish QR recognition and intermittent head measurement, but no completed joint observation.

Two concrete causes explain the low admitted-angle rate:

1. **The head/neck junction test is fragile at this image scale.** Head height is around 73–77 pixels. The measured paper inset and two-pixel uncertainty allowance reduce the permitted neck-start gap to **one pixel** for 185 of the 186 processed results. Rejected gaps range from two to six pixels. [head_model_neck.py](../../scripts/aufgabe04/perception/stand_axis/head_model_neck.py), lines 91–117, searches for uninterrupted simultaneous edge pixels at two fixed columns over roughly 9–10 rows. Its reported gap is the delay until that run begins, not a direct physical measurement of a detached neck. Border selection, slanted or fragmented raw rails, and quantization need replay-based discrimination; increasing the gap threshold alone risks accepting an inner paper border as the physical head.
2. **A successful primary fit skips the registration path needed to associate it.** [camera_target_registration.py](../../scripts/aufgabe04/real_robot/observer/camera_target_registration.py), line 228, returns immediately when the primary estimate is usable. The later association branch in [node.py](../../scripts/aufgabe04/real_robot/observer/node.py), lines 1789–1844, uses camera-centered association only for a registered ROI. These 24 valid primary fits therefore encounter the original map-centered **3-degree half-cone**, while their measured head rays differ from the projected bearing by **3.305–3.483 degrees**. Frames that enter bounded reacquisition can register their observed center and pass the same narrow cone at that center. This makes admission depend on which acquisition branch found the head.

An association-only replay of the **eight rejected usable fits with saved synchronized scans** reproduces all eight original map-cone rejections. Applying the existing bounded camera-registered helper to the recorded head rays admits **seven**, while one remains rejected for multiple eligible clusters. The replay preserves the existing range, narrow cone, freshness and uniqueness rules. It does not create registration receipts or replay QR binding, consensus and recommendation end to end; it therefore does not prove the first observation would have completed. The 64-frame capture limit means most later frames have decision logs but no saved full sensor tuple.

## Why the two additional inspection requests?

There is no rule requiring two extra inspections. [inspection_execution.py](../../scripts/aufgabe04/real_robot/candidate/inspection_execution.py), lines 75–82, returns immediately if a recommendation exists. An inspection-progress artifact does not satisfy that completion condition.

| Stage | Recorded decision |
| --- | --- |
| First point: `front_readable`, no admitted advisory yaw | The generic view policy requests **+45 degrees** around the candidate. `inspection_001` completes at **14:00:10.356 UTC**. |
| Second camera point, **14:00:16.716–14:01:46.260 UTC** | The 90-second observer deadline expires with **zero associated frames, zero angle samples and no QR identity**. The controller records `unobservable`. |
| Next local-view search | The +90-degree direction fails static route planning at eight bounded standoffs. The -90-degree alternative is selected. `inspection_002` passes dry admission at **14:03:53.629 UTC**; the snapshot contains no completed execution or third camera observation. |

The first additional view also visibly contains the front face and complete QR in the full image:

![Second camera point, original saved frame](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/results/aufgabe04/real/autonomous_exploration/stand_explore_exact2_camera_all5_20260911T135346Z/candidates/000_survey_candidate_0003/camera_lidar_attempt_01/perception_debug/latest_frame.png)

Its detector loses the target through another admission/crop interaction. In the final saved metadata, the visible head spans approximately **x=311–382**, while the nominal crop begins at **x=337**, clipping the left head/QR border. Bounded wider acquisition does locate the whole head. However, **all 136 fresh processed results** reject its LiDAR association as `no_samples_in_accepted_range`, preventing the recentered strict measurement. The allowed range ends at **0.736848 m**; nearest returns in the camera cone are **0.741–0.759 m**, exceeding that bound by **4.15–22.15 mm**. The logs do not independently establish the physical cause of that range discrepancy. Another 63 of 199 processed results are obsolete. Thus the generic `unobservable` outcome hides a clearly visible head whose candidate association failed.

The configured ceiling is eight observed camera views, not a required tour length. The missing joint recommendation drives continued search. The snapshot contains two observed camera points, one completed additional move, and a second additional move prepared through dry admission.

## Recommended correction and evidence limits

The immediate code correction is to apply the same bounded camera/head-to-candidate registration contract to an already valid measured head fit, with explicit provenance and the existing unique-cluster/range gates. Successful primary geometry must be able to obtain the association currently available only after acquisition failure. Replay all saved cases, including the ambiguous cluster, before changing live behavior.

Next, make neck-support verification follow the observed raw rail geometry and validate head-versus-paper boundary selection on these original frames, retaining rejection of paper-only proposals. Investigate the second view's projected range and candidate/scan consistency before considering any bounded range recovery. The controller should preserve a specific visible-head/association failure reason instead of reducing it to `unobservable`; that distinction can guide stopped recovery and avoid unhelpful view changes. Lowering sample counts, broadly enlarging ranges, or promoting a purple prediction would bypass the evidence problem.

The existing measured model is already active. This run supports work on its admission and handoff behavior; it does not establish physical angle accuracy, a camera hardware fault, or end-to-end success from the proposed correction.

The [audit script](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/audit_camera_decisions.py), [decision summary](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/camera_decision_audit.json), [association counterfactual](../../results/audits/stand_explore_exact2_camera_all5_20260911T135346Z/bounded_association_counterfactual.json), original source archive and hash manifest are preserved. All 619 original files remain unchanged. This audit changed no production code and started no ROS nodes or robot motion.
