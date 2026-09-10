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
