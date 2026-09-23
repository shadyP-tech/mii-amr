"""Optional stopped facing validation after successful opposite-side discovery."""
from dataclasses import replace

from scripts.aufgabe04.artifacts.retained_facing import build_retained_facing
from scripts.aufgabe04.navigation.approach.viewpoint_recommendation import recommendation_to_dict
from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json


def try_retained_facing(*, observation, discovery, frame, effects, output_dir):
    """A rejected facing route must never revoke already admitted identity."""
    if not discovery.get('retained_backside_orientation'):
        return None
    from scripts.aufgabe04.real_robot.candidate.approach import FacingValidationRequest, CandidateApproachPoseError, _write_json, _read_finite_pose2d
    path = output_dir/'retained_facing_recommendation.json'
    diagnostic = dict(policy='retained_backside_current_qr_facing', motion_authorized=False,
                      current_angle_refit=False, qr_discovery_preserved=True)
    try:
        recommendation = build_retained_facing(observation.qr_observation_pose_path,
            stand_radius_m=frame.candidate.geometry.radius_m,
            target_distance_m=frame.config.final_facing_offset_m)
        _write_json(path, recommendation_to_dict(recommendation))
        pose = _read_finite_pose2d(effects, context='retained_facing_validation',
                                 candidate_uid=frame.candidate.candidate_uid)
        facing = dict(effects.validate_facing(FacingValidationRequest(
            config=frame.config, candidate=frame.candidate, recommendation_path=path,
            current_pose=pose, output_dir=output_dir/'retained_facing_validation')))
    except (ValueError, OSError, CandidateApproachPoseError) as exc:
        diagnostic.update(facing_ready=False, reason=str(exc))
        write_content_hashed_json(output_dir/'retained_facing_status.json', diagnostic,
                                 hash_field='retained_facing_status_sha256')
        return None
    facing.update(qr_id=observation.qr_id, facing_ready=True,
                  validated_target_center=discovery['retained_backside_orientation']['validated_target_center'],
                  retained_qr_observation_pose_json=str(observation.qr_observation_pose_path))
    diagnostic.update(facing_ready=True, recommendation_json=str(path))
    write_content_hashed_json(output_dir/'retained_facing_status.json', diagnostic,
                             hash_field='retained_facing_status_sha256')
    return replace(observation, recommendation_path=path, qr_observation_pose_path=None), facing
