"""Persist candidate-local target deferral without changing obstacle keepouts."""

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

from scripts.aufgabe04.artifacts.content_store import (
    load_content_hashed_json, payload_sha256, write_content_hashed_json,
)
from scripts.aufgabe04.artifacts.current_target_estimate import planning_target_geometry
from scripts.aufgabe04.navigation.approach.candidate_frame_projection import CandidatePlanningFrame
from scripts.aufgabe04.navigation.approach.backside_axis_frame_projection import (
    load_backside_axis_planning_observation,
)
from scripts.aufgabe04.navigation.approach.candidate_target_admission import (
    CandidateTargetAdmission, evaluate_candidate_target_admission, load_candidate_target_costmap,
)
from scripts.aufgabe04.navigation.coverage.stand_candidate_population_retention import classify_static_map_population_retention
from scripts.aufgabe04.navigation.coverage.stand_candidate_static_map_admission import (
    STATIC_MAP_CLEARANCE_BELOW_REQUIRED, StandCandidateStaticMapEvidence,
)
from scripts.aufgabe04.navigation.coverage.stand_coverage_survey import coverage_survey_plan_sha256
from scripts.aufgabe04.artifacts.candidate_perception_advisory import MORPHOLOGY_CONFLICT
from scripts.aufgabe04.real_robot.candidate.observation_deferral import (
    CandidateObservationUnavailableError,
)
from scripts.aufgabe04.real_robot.candidate.current_lidar_targets import (
    HASH_FIELD as LIDAR_TARGET_HASH_FIELD, POLICY as LIDAR_TARGET_POLICY,
    load_current_lidar_target,
)
from scripts.aufgabe04.stations.candidate_snapshot import (
    CandidateSnapshot, candidate_geometry_sha256, candidate_snapshot_sha256,
    validate_candidate_geometry, validate_candidate_snapshot,
)


@dataclass(frozen=True)
class RetainedLidarTargetBinding:
    """Original stopped support; subsequent localization never rewrites it."""

    candidate_uid: str
    evidence_path: Path
    evidence_sha256: str
    source_snapshot: CandidateSnapshot
    source_snapshot_sha256: str
    source_planning_frame: CandidatePlanningFrame


SURVEY_OBSERVATION_POLICY = "retained_survey_observation_only"
SURVEY_OBSERVATION_HASH_FIELD = "survey_observation_support_sha256"
_SURVEY_OBSERVATION_FIELDS = frozenset({
    "schema_version", "policy", "source_kind", "candidate_uid",
    "candidate_snapshot_sha256", "candidate_geometry_sha256", "map_bundle_sha256",
    "coverage_plan_sha256", "planning_frame", "target_admission", "purpose",
    "precise_motion_authorized", "motion_authorized", "keepouts_changed",
})


@dataclass(frozen=True)
class RetainedSurveyTargetBinding:
    """A bounded survey point, never a fitted head or precise motion target."""

    candidate_uid: str
    evidence_path: Path
    evidence_sha256: str
    source_snapshot: CandidateSnapshot
    source_snapshot_sha256: str
    source_planning_frame: CandidatePlanningFrame


def write_survey_observation_support(path, *, config, planning_frame, candidate_uid):
    """Bind a surveyed hypothesis to a stopped observation approach, without scans."""
    validate_candidate_snapshot(config.snapshot)
    candidate = config.snapshot.candidate_for(candidate_uid)
    if (candidate is None or not isinstance(planning_frame, CandidatePlanningFrame)
            or planning_frame.map_frame != config.snapshot.planning_frame):
        raise ValueError("survey observation candidate/planning frame mismatch")
    decision = evaluate_target(config, candidate)
    if not decision.accepted:
        raise CandidateObservationUnavailableError(
            candidate_uid=candidate_uid, observation_attempt_index=0,
            reason="candidate_target_ineligible",
            process_evidence={"observer_started": False, "motion_authorized": False},
            status_evidence=decision.to_evidence(),
        )
    evidence = {
        "schema_version": 1, "policy": SURVEY_OBSERVATION_POLICY,
        "source_kind": "survey_candidate_snapshot", "candidate_uid": candidate_uid,
        "candidate_snapshot_sha256": candidate_snapshot_sha256(config.snapshot),
        "candidate_geometry_sha256": candidate_geometry_sha256(candidate.geometry),
        "map_bundle_sha256": config.snapshot.map_bundle_sha256,
        "coverage_plan_sha256": coverage_survey_plan_sha256(config.plan),
        "planning_frame": planning_frame.to_evidence(), "target_admission": decision.to_evidence(),
        "purpose": "observation_only", "precise_motion_authorized": False,
        "motion_authorized": False, "keepouts_changed": False,
    }
    write_content_hashed_json(path, evidence, hash_field=SURVEY_OBSERVATION_HASH_FIELD)
    return load_survey_observation_support(path, candidate_uid=candidate_uid, snapshot=config.snapshot)


def _survey_admission_from_clearance(candidate, clearance):
    """Check the saved decision's internal contract; live consumers recheck the map."""
    if type(clearance) not in (int, float):
        raise ValueError("survey observation static clearance must be numeric")
    geometry = candidate.geometry
    retention = classify_static_map_population_retention(clearance_m=clearance,
        candidate_radius_m=geometry.radius_m, candidate_uncertainty_m=geometry.uncertainty_m)
    static = StandCandidateStaticMapEvidence(
        stand_id=candidate.candidate_uid, x_m=geometry.x_m, y_m=geometry.y_m,
        confidence=candidate.confidence, hit_count=candidate.hit_count,
        source_observation_ids=candidate.source.observation_ids, static_map_clearance_m=clearance,
        required_clearance_m=geometry.radius_m+geometry.uncertainty_m,
        clearance_shortfall_m=retention.clearance_shortfall_m, disposition=retention.disposition,
        population_retained=retention.population_retained, boundary_provisional=retention.boundary_provisional,
        admitted=retention.strictly_admitted,
        reasons=() if retention.strictly_admitted else (STATIC_MAP_CLEARANCE_BELOW_REQUIRED,),
    )
    if (not retention.population_retained
            or any(item.kind == MORPHOLOGY_CONFLICT for item in candidate.source.perception_advisories)):
        raise ValueError("survey observation target admission is rejected")
    geometry_hash = candidate_geometry_sha256(geometry)
    return CandidateTargetAdmission(candidate.candidate_uid, True, (), geometry_hash,
        geometry_hash, static, ()).to_evidence()


def load_survey_observation_support(path, *, candidate_uid, snapshot):
    """Read a scan-free observation proof bound to one immutable survey snapshot."""
    validate_candidate_snapshot(snapshot)
    evidence = load_content_hashed_json(path, hash_field=SURVEY_OBSERVATION_HASH_FIELD)
    candidate = snapshot.candidate_for(candidate_uid)
    if (set(evidence) != _SURVEY_OBSERVATION_FIELDS
            or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1
            or evidence.get("policy") != SURVEY_OBSERVATION_POLICY
            or evidence.get("source_kind") != "survey_candidate_snapshot"
            or evidence.get("purpose") != "observation_only"
            or any(evidence.get(key) is not False for key in (
                "precise_motion_authorized", "motion_authorized", "keepouts_changed"))
            or candidate is None or evidence.get("candidate_uid") != candidate_uid
            or evidence.get("candidate_snapshot_sha256") != candidate_snapshot_sha256(snapshot)
            or evidence.get("map_bundle_sha256") != snapshot.map_bundle_sha256
            or evidence.get("candidate_geometry_sha256") != candidate_geometry_sha256(candidate.geometry)):
        raise ValueError("survey observation proof candidate/policy/snapshot mismatch")
    plan_hash = evidence["coverage_plan_sha256"]
    if (not isinstance(plan_hash, str) or len(plan_hash) != 64
            or any(char not in "0123456789abcdef" for char in plan_hash)):
        raise ValueError("survey observation coverage plan binding is invalid")
    planning = CandidatePlanningFrame.from_evidence(evidence["planning_frame"])
    if planning.map_frame != snapshot.planning_frame:
        raise ValueError("survey observation planning frame differs from snapshot")
    decision = evidence["target_admission"]
    try:
        expected = _survey_admission_from_clearance(candidate,
            decision["static_map_evidence"]["static_map_clearance_m"])
    except (KeyError, TypeError) as exc:
        raise ValueError("survey observation target admission is malformed") from exc
    if payload_sha256(decision) != payload_sha256(expected):
        raise ValueError("survey observation target admission differs from bound candidate")
    return evidence


def _project_geometry(geometry, before, after):
    a, b = before.map_from_odom, after.map_from_odom
    if a == b:
        return geometry
    dx, dy = geometry.x_m-a.x_m, geometry.y_m-a.y_m
    ox = math.cos(a.yaw_rad)*dx + math.sin(a.yaw_rad)*dy
    oy = -math.sin(a.yaw_rad)*dx + math.cos(a.yaw_rad)*dy
    return replace(geometry,
        x_m=b.x_m+math.cos(b.yaw_rad)*ox-math.sin(b.yaw_rad)*oy,
        y_m=b.y_m+math.sin(b.yaw_rad)*ox+math.cos(b.yaw_rad)*oy)


def _load_retained_lidar_geometry(binding):
    if not isinstance(binding, RetainedLidarTargetBinding):
        raise ValueError("invalid retained LiDAR target binding")
    proof = load_content_hashed_json(binding.evidence_path, hash_field=LIDAR_TARGET_HASH_FIELD)
    if (payload_sha256(proof) != binding.evidence_sha256
            or candidate_snapshot_sha256(binding.source_snapshot) != binding.source_snapshot_sha256
            or proof.get("planning_frame") != binding.source_planning_frame.to_evidence()):
        raise ValueError("retained LiDAR target original source binding changed")
    estimate = load_current_lidar_target(binding.evidence_path,
        candidate_uid=binding.candidate_uid, snapshot=binding.source_snapshot)
    candidate = binding.source_snapshot.candidate_for(binding.candidate_uid)
    if candidate is None or estimate.get("policy") != LIDAR_TARGET_POLICY:
        raise ValueError("retained LiDAR target candidate/policy mismatch")
    geometry = planning_target_geometry(candidate, estimate)
    limit = min(.16, 2*(candidate.geometry.radius_m+candidate.geometry.uncertainty_m))
    if (geometry.uncertainty_m > .14
            or math.hypot(geometry.x_m-candidate.geometry.x_m,
                          geometry.y_m-candidate.geometry.y_m) > limit+1e-9):
        raise ValueError("retained LiDAR target outside original support bound")
    return geometry


def _lidar_geometry_in_frame(binding, geometry, frame):
    original = binding.source_snapshot
    snapshot, planning = frame.config.snapshot, frame.planning_frame
    if (planning is None or binding.candidate_uid != frame.candidate.candidate_uid
            or snapshot.candidate_for(binding.candidate_uid) != frame.candidate
            or replace(snapshot, candidates=original.candidates) != original
            or (planning.map_frame, planning.odom_frame) != (
                binding.source_planning_frame.map_frame, binding.source_planning_frame.odom_frame)
            or planning.map_frame != snapshot.planning_frame):
        raise ValueError("retained LiDAR target candidate/snapshot/frame mismatch")
    # Every obstacle remains the same canonical odom obstacle. Only its map
    # coordinates may change when stationary localization is refreshed.
    if len(snapshot.candidates) != len(original.candidates):
        raise ValueError("retained LiDAR target snapshot population changed")
    for before, after in zip(original.candidates, snapshot.candidates):
        expected = _project_geometry(before.geometry, binding.source_planning_frame, planning)
        if (replace(after, geometry=before.geometry) != before
                or any(abs(getattr(after.geometry, key)-getattr(expected, key)) > 1e-9
                       for key in asdict(expected))):
            raise ValueError("retained LiDAR target snapshot geometry/projection mismatch")
    return _project_geometry(geometry, binding.source_planning_frame, planning)


def bind_current_lidar_target(frame, *, evidence_path: Path):
    """Bind one accepted preapproach point to its immutable acquisition source."""
    proof = load_content_hashed_json(evidence_path, hash_field=LIDAR_TARGET_HASH_FIELD)
    if frame.planning_frame is None:
        raise ValueError("current LiDAR target requires an admitted planning frame")
    binding = RetainedLidarTargetBinding(frame.candidate.candidate_uid, Path(evidence_path),
        payload_sha256(proof), frame.config.snapshot,
        candidate_snapshot_sha256(frame.config.snapshot), frame.planning_frame)
    geometry = _load_retained_lidar_geometry(binding)
    target = _lidar_geometry_in_frame(binding, geometry, frame)
    existing = getattr(frame, "camera_target_geometry", None)
    if existing is not None and existing != target:
        raise ValueError("current LiDAR target differs from selected camera target")
    changes = {"retained_survey_target": None} if hasattr(frame, "retained_survey_target") else {}
    return replace(frame, **changes, camera_target_geometry=target, camera_alignment=None,
                   current_lidar_target_path=Path(evidence_path), retained_lidar_target=binding)


def _require_retained_lidar_geometry(frame):
    binding = frame.retained_lidar_target
    original_geometry = _load_retained_lidar_geometry(binding)
    expected = _lidar_geometry_in_frame(binding, original_geometry, frame)
    if (getattr(frame, "camera_target_geometry", None) != expected
            or getattr(frame, "current_lidar_target_path", None) != binding.evidence_path):
        raise ValueError("retained LiDAR target geometry/path binding changed")
    return original_geometry


def _load_retained_survey_geometry(binding, *, plan=None):
    if not isinstance(binding, RetainedSurveyTargetBinding):
        raise ValueError("invalid retained survey target binding")
    proof = load_survey_observation_support(binding.evidence_path,
        candidate_uid=binding.candidate_uid, snapshot=binding.source_snapshot)
    if (payload_sha256(proof) != binding.evidence_sha256
            or candidate_snapshot_sha256(binding.source_snapshot) != binding.source_snapshot_sha256
            or proof["planning_frame"] != binding.source_planning_frame.to_evidence()
            or plan is not None and proof["coverage_plan_sha256"] != coverage_survey_plan_sha256(plan)):
        raise ValueError("retained survey target original source binding changed")
    return binding.source_snapshot.candidate_for(binding.candidate_uid).geometry


def bind_survey_observation_target(frame, *, evidence_path: Path):
    """Retain the actual survey target of a completed observation-only route."""
    if frame.planning_frame is None:
        raise ValueError("survey observation requires an admitted planning frame")
    proof = load_survey_observation_support(evidence_path,
        candidate_uid=frame.candidate.candidate_uid, snapshot=frame.config.snapshot)
    if proof["coverage_plan_sha256"] != coverage_survey_plan_sha256(frame.config.plan):
        raise ValueError("survey observation coverage plan binding changed")
    binding = RetainedSurveyTargetBinding(frame.candidate.candidate_uid, Path(evidence_path),
        payload_sha256(proof), frame.config.snapshot,
        candidate_snapshot_sha256(frame.config.snapshot), frame.planning_frame)
    geometry = _load_retained_survey_geometry(binding, plan=frame.config.plan)
    target = _lidar_geometry_in_frame(binding, geometry, frame)
    if frame.camera_target_geometry is not None and frame.camera_target_geometry != target:
        raise ValueError("survey observation differs from selected camera target")
    return replace(frame, camera_target_geometry=target, camera_alignment=None,
        current_lidar_target_path=None, retained_lidar_target=None, retained_survey_target=binding)


def _require_retained_survey_geometry(frame):
    binding = frame.retained_survey_target
    original = _load_retained_survey_geometry(binding, plan=frame.config.plan)
    expected = _lidar_geometry_in_frame(binding, original, frame)
    if (frame.camera_target_geometry != expected
            or frame.current_lidar_target_path is not None
            or frame.retained_lidar_target is not None):
        raise ValueError("retained survey target geometry/provenance changed")
    return original


def retain_camera_target_geometry(source, arrival, *, evidence_path: Path,
                                  alignment_uncertainty=None):
    """Carry a bound fitted point through odom into one newly admitted frame.

    Current verified geometry supersedes a prior selected alignment. This
    retains a point and its intrinsic uncertainty; it never retains a claim
    that the camera is aligned after moving or refreshing localization.
    """
    geometry = getattr(source, "camera_target_geometry", None)
    lidar_binding = getattr(source, "retained_lidar_target", None)
    survey_binding = getattr(source, "retained_survey_target", None)
    alignment = getattr(source, "camera_alignment", None)
    if geometry is None and alignment is None and lidar_binding is None and survey_binding is None:
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
    if survey_binding is not None:
        original_geometry = _require_retained_survey_geometry(source)
        if coverage_survey_plan_sha256(source.config.plan) != coverage_survey_plan_sha256(arrival.config.plan):
            raise ValueError("retained survey target coverage plan binding changed")
        target = _lidar_geometry_in_frame(survey_binding, original_geometry, arrival)
        projected_alignment = None
        kind = SURVEY_OBSERVATION_POLICY
    elif lidar_binding is not None:
        original_geometry = _require_retained_lidar_geometry(source)
        target = _lidar_geometry_in_frame(lidar_binding, original_geometry, arrival)
        projected_alignment = None
        kind = LIDAR_TARGET_POLICY
    elif geometry is None:
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
        projected = _project_geometry(geometry, before, after)
        target = replace(arrival.candidate.geometry, x_m=projected.x_m,
            y_m=projected.y_m, uncertainty_m=geometry.uncertainty_m)
        projected_alignment = None
        kind = "current_camera_target_geometry"
    for frame, point in ((source, geometry), (arrival, target)):
        validate_candidate_geometry(point)
        envelope = frame.candidate.geometry
        limit = envelope.radius_m+envelope.uncertainty_m
        if lidar_binding is not None:
            limit = min(.16, 2*limit)
        if math.hypot(point.x_m-envelope.x_m, point.y_m-envelope.y_m) > limit+1e-9:
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
    if lidar_binding is not None:
        provenance["retained_lidar_target"] = {
            "evidence_path": str(lidar_binding.evidence_path),
            "evidence_sha256": lidar_binding.evidence_sha256,
            "source_candidate_snapshot_sha256": lidar_binding.source_snapshot_sha256,
            "source_planning_frame": lidar_binding.source_planning_frame.to_evidence(),
        }
    if survey_binding is not None:
        provenance["retained_survey_target"] = {
            "evidence_path": str(survey_binding.evidence_path),
            "evidence_sha256": survey_binding.evidence_sha256,
            "source_candidate_snapshot_sha256": survey_binding.source_snapshot_sha256,
            "source_planning_frame": survey_binding.source_planning_frame.to_evidence(),
            "purpose": "observation_only", "precise_motion_authorized": False,
        }
    write_content_hashed_json(evidence_path, provenance,
        hash_field="camera_target_geometry_projection_sha256")
    changes = {} if lidar_binding is None else dict(retained_lidar_target=lidar_binding,
        current_lidar_target_path=lidar_binding.evidence_path)
    if survey_binding is not None:
        changes.update(retained_survey_target=survey_binding,
                       retained_lidar_target=None, current_lidar_target_path=None)
    return replace(arrival, **changes, camera_target_geometry=target,
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
    if getattr(frame, "retained_lidar_target", None) is not None:
        _require_retained_lidar_geometry(frame)
    if getattr(frame, "retained_survey_target", None) is not None:
        _require_retained_survey_geometry(frame)
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
