"""Decision fields from the recorded second-candidate camera deadline.

Run: stand_explore_exact2_camera_all5_20260910T140546Z
Candidate: 001_survey_candidate_0001, camera_lidar_attempt_01
Source observer_status.json SHA256:
5c08e3ba0bfb762936551ca99a0f9eacfa601b59cd28f058f4a99db0de193e13

No images, stale poses or motion authority are included in this fixture.
"""


def recorded_backside_timeout_status() -> dict[str, object]:
    return {
        "state": "obsolete_detector_result",
        "image_age_sec": 0.5087933540344238,
        "axis_consensus": {
            "sample_count": 0,
            "peak_sample_count": 0,
            "required_sample_count": 7,
        },
        "observation_evidence": {
            "accepted_frame_count": 0,
            "lidar_rejection_count": 21,
            "soft_miss_count": 208,
            "last_soft_miss_reason": "obsolete_detector_result",
            "poisoned": False,
            "poison_reason": None,
        },
        "tf_retry_attempt_summary": {
            "attempted_tuple_count": 81,
            "exhausted_tuple_count": 14,
            "peak_retry_count": 10,
        },
    }


def recorded_front_timeout_status(event_line: int = 408) -> dict[str, object]:
    """Decision fields from the final two 2026-09-14 front-view events.

    Run: stand_explore_exact2_camera_all5_20260914T123717Z
    Candidate: 001_survey_candidate_0001, camera_lidar_attempt_02
    Original observer_events.jsonl SHA256:
    e3c027d1781c333bdfc9b6e98dbde6e29175b0d543e2c5e5af4bf80d2c8a2f66
    Original observer_status.json (event 408) SHA256:
    dee168b0396429193e29b7fc6b30544e7176188493b8f8c4837ac0d7bab3ee30
    """

    if event_line not in (407, 408):
        raise ValueError("recorded front deadline has event lines 407 and 408")
    stale_input = event_line == 408
    state = "stale_sensor_tuple" if stale_input else "obsolete_detector_result"
    return {
        "state": state,
        "observed_unix_sec": 1789390171.47338 if stale_input else 1789390171.470201,
        "image_stamp_sec": 1789390170.838845 if stale_input else 1789390170.738862,
        "image_age_sec": 0.6341516971588135 if stale_input else 0.7167012691497803,
        **({"transient_tf_retry": False} if stale_input else {}),
        "axis_consensus": {
            "sample_count": 0, "peak_sample_count": 0, "required_sample_count": 7,
        },
        "observation_evidence": {
            "accepted_frame_count": 0, "lidar_rejection_count": 23,
            "soft_miss_count": 125 if stale_input else 124,
            "last_soft_miss_reason": state, "poisoned": False, "poison_reason": None,
        },
        "tf_retry": {"retry_count": 0 if stale_input else 6},
        "tf_retry_attempt_summary": {
            "attempted_tuple_count": 140, "exhausted_tuple_count": 118, "peak_retry_count": 10,
        },
    }
