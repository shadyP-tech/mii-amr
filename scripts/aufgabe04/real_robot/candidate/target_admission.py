"""Persist candidate-local target deferral without changing obstacle keepouts."""

from dataclasses import asdict, replace
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import write_content_hashed_json
from scripts.aufgabe04.artifacts.current_target_estimate import planning_target_geometry
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    load_backside_axis_planning_observation,
)
from scripts.aufgabe04.navigation.approach.candidate_target_admission import (
    evaluate_candidate_target_admission, load_candidate_target_costmap,
)
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    candidate_geometry_sha256, candidate_snapshot_sha256, validate_candidate_geometry,
)


def retain_camera_target_geometry(source, arrival, *, evidence_path: Path,
                                  alignment_uncertainty=None):
    """Carry a bound fitted point through odom into one newly admitted frame.

    Current verified geometry supersedes a prior selected alignment. This
    retains a point and its intrinsic uncertainty; it never retains a claim
    that the camera is aligned after moving or refreshing localization.
    """
    geometry = getattr(source, "camera_target_geometry", None)
    alignment = getattr(source, "camera_alignment", None)
    if geometry is None and alignment is None:
        return arrival
    if getattr(arrival, "camera_target_geometry", None) is not None:
        return arrival
    uid = source.candidate.candidate_uid
    source_hash = candidate_snapshot_sha256(source.config.snapshot)
    target_hash = candidate_snapshot_sha256(arrival.config.snapshot)
    if (source.config.snapshot.candidate_for(uid) != source.candidate
            or arrival.candidate.candidate_uid != uid
            or arrival.config.snapshot.candidate_for(uid) != arrival.candidate
            or source.config.snapshot.map_bundle_sha256 != arrival.config.snapshot.map_bundle_sha256):
        raise ValueError("retained camera target candidate/snapshot binding mismatch")
    before, after = source.planning_frame, arrival.planning_frame
    if before is None or after is None or (
        before.map_frame, before.odom_frame
    ) != (after.map_frame, after.odom_frame):
        raise ValueError("retained camera target requires matching admitted map/odom frames")
    if (before.map_frame != source.config.snapshot.planning_frame
            or after.map_frame != arrival.config.snapshot.planning_frame):
        raise ValueError("retained camera target planning frame mismatch")
    if geometry is None:
        from scripts.aufgabe04.navigation.approach.camera_head_alignment import (
            reproject_camera_alignment, validate_camera_alignment,
        )
        from scripts.aufgabe04.real_robot.configuration.profile import camera_calibration_sha256
        validate_camera_alignment(alignment)
        if (alignment["candidate_uid"] != uid
                or alignment["candidate_snapshot_sha256"] != source_hash
                or source.config.camera_calibration is None
                or arrival.config.camera_calibration is None
                or alignment["camera_calibration_sha256"] != camera_calibration_sha256(source.config.camera_calibration)
                or alignment["camera_calibration_sha256"] != camera_calibration_sha256(arrival.config.camera_calibration)):
            raise ValueError("retained camera alignment candidate/snapshot/calibration binding mismatch")
        geometry = replace(source.candidate.geometry,
            x_m=alignment["center_x_m"], y_m=alignment["center_y_m"],
            uncertainty_m=alignment["center_uncertainty_m"])
        projected_alignment = reproject_camera_alignment(
            alignment, snapshot=arrival.config.snapshot, candidate_uid=uid,
            source_map_from_odom=before.map_from_odom,
            target_map_from_odom=after.map_from_odom,
            uncertainty=alignment_uncertainty,
        )
        target = replace(arrival.candidate.geometry,
            x_m=projected_alignment["center_x_m"], y_m=projected_alignment["center_y_m"],
            uncertainty_m=projected_alignment["center_uncertainty_m"])
        kind = "selected_camera_alignment"
    else:
        # The fit lives in the source frame, not the refreshed map frame.
        # Rotate/translate via its canonical odom point; never copy x/y.
        a, b = before.map_from_odom, after.map_from_odom
        dx, dy = geometry.x_m-a.x_m, geometry.y_m-a.y_m
        ox = math.cos(a.yaw_rad)*dx + math.sin(a.yaw_rad)*dy
        oy = -math.sin(a.yaw_rad)*dx + math.cos(a.yaw_rad)*dy
        target = replace(arrival.candidate.geometry,
            x_m=b.x_m+math.cos(b.yaw_rad)*ox-math.sin(b.yaw_rad)*oy,
            y_m=b.y_m+math.sin(b.yaw_rad)*ox+math.cos(b.yaw_rad)*oy,
            uncertainty_m=geometry.uncertainty_m)
        projected_alignment = None
        kind = "current_camera_target_geometry"
    for frame, point in ((source, geometry), (arrival, target)):
        validate_candidate_geometry(point)
        envelope = frame.candidate.geometry
        if math.hypot(point.x_m-envelope.x_m, point.y_m-envelope.y_m) > envelope.radius_m+envelope.uncertainty_m+1e-9:
            raise ValueError("retained fitted camera target outside candidate envelope")
    provenance = {
        "schema_version": 1, "candidate_uid": uid, "source_kind": kind,
        "source_candidate_snapshot_sha256": source_hash,
        "target_candidate_snapshot_sha256": target_hash,
        "source_map_from_odom": asdict(before.map_from_odom),
        "target_map_from_odom": asdict(after.map_from_odom),
        "source_target_geometry": asdict(geometry),
        "source_target_geometry_sha256": candidate_geometry_sha256(geometry),
        "target_geometry": asdict(target),
        "target_geometry_sha256": candidate_geometry_sha256(target),
        "camera_alignment": projected_alignment,
        "head_alignment_verified": False, "camera_centered": False,
        "motion_authorized": False, "keepouts_changed": False,
    }
    write_content_hashed_json(evidence_path, provenance,
        hash_field="camera_target_geometry_projection_sha256")
    return replace(arrival, camera_target_geometry=target,
                   camera_alignment=projected_alignment,
                   camera_target_geometry_evidence_path=evidence_path)


def evaluate_target(config, candidate, *, target_geometry=None):
    """Check the current target against the bound static map and arena."""
    costmap = load_candidate_target_costmap(
        config.map_yaml, semantic_map_id=config.semantic_map_id,
        plan=config.plan, snapshot=config.snapshot,
    )
    return evaluate_candidate_target_admission(
        candidate, costmap, target_geometry=target_geometry,
    )


def require_target(config, candidate, *, evidence_path: Path, attempt_index: int,
                   target_geometry=None):
    """Persist both acceptance and deferral; only a caller can authorize motion."""
    decision = evaluate_target(config, candidate, target_geometry=target_geometry)
    evidence = {
        **decision.to_evidence(),
        "map_bundle_sha256": config.snapshot.map_bundle_sha256,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(config.snapshot),
        "keepouts_changed": False,
    }
    digest = write_content_hashed_json(
        evidence_path, evidence, hash_field="candidate_target_admission_sha256",
    )
    if not decision.accepted:
        raise CandidateObservationUnavailableError(
            candidate_uid=candidate.candidate_uid,
            observation_attempt_index=attempt_index,
            reason="candidate_target_ineligible",
            process_evidence={"observer_started": False, "motion_authorized": False,
                              "candidate_target_admission_path": str(evidence_path),
                              "candidate_target_admission_sha256": digest},
            status_evidence=evidence,
        )
    return decision


def require_frame_target(frame, *, evidence_path: Path, attempt_index: int):
    """Resolve retained geometry only after it has been projected to this frame."""
    geometry = getattr(frame, "camera_target_geometry", None)
    retained = getattr(frame, "retained_backside_axis_path", None)
    if retained is not None:
        observation = load_backside_axis_planning_observation(retained)
        if (observation.stand_id != frame.candidate.candidate_uid
                or observation.planning_frame != frame.config.planning_frame
                or abs(observation.stand_x_m - frame.candidate.geometry.x_m) > 1e-6
                or abs(observation.stand_y_m - frame.candidate.geometry.y_m) > 1e-6):
            raise ValueError("retained target admission candidate mismatch")
        geometry = planning_target_geometry(frame.candidate, observation.validated_target_center)
    return require_target(
        frame.config, frame.candidate, evidence_path=evidence_path,
        attempt_index=attempt_index, target_geometry=geometry,
    )
