"""Navigation compatibility facade for backside-axis artifact binding.

The authoritative ROS-free schema and geometry live in the artifact layer so
the passive observer and navigation cannot drift independently. Existing
imports remain stable through this facade.
"""

from scripts.aufgabe04.artifacts.backside_axis_observation import (
    BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION,
    BACKSIDE_CLASSIFICATION_BASIS,
    BACKSIDE_CURRENT_FRAME_SOURCE,
    BACKSIDE_MODEL_EVIDENCE_STATE,
    BACKSIDE_VISIBLE_FACE,
    BacksideAxisObservation,
    MAXIMUM_HEAD_CENTER_ERROR_RATIO,
    MAXIMUM_HEAD_SCALE_RATIO,
    MINIMUM_BACKSIDE_AXIS_CONFIDENCE,
    MINIMUM_BACKSIDE_AXIS_SAMPLE_COUNT,
    MINIMUM_BACKSIDE_FACE_CONFIDENCE,
    MINIMUM_HEAD_SCALE_RATIO,
    PASSIVE_VIEWPOINT_OBSERVER_VERSION,
    REAL_STAND_AXIS_OBSERVATION_KIND,
    load_backside_axis_observation,
    load_opposite_face_normal,
    opposite_face_normal_from_axis_observation,
    validated_backside_axis_observation,
)


__all__ = [
    "BACKSIDE_AXIS_OBSERVATION_SCHEMA_VERSION",
    "BACKSIDE_CLASSIFICATION_BASIS",
    "BACKSIDE_CURRENT_FRAME_SOURCE",
    "BACKSIDE_MODEL_EVIDENCE_STATE",
    "BACKSIDE_VISIBLE_FACE",
    "BacksideAxisObservation",
    "MAXIMUM_HEAD_CENTER_ERROR_RATIO",
    "MAXIMUM_HEAD_SCALE_RATIO",
    "MINIMUM_BACKSIDE_AXIS_CONFIDENCE",
    "MINIMUM_BACKSIDE_AXIS_SAMPLE_COUNT",
    "MINIMUM_BACKSIDE_FACE_CONFIDENCE",
    "MINIMUM_HEAD_SCALE_RATIO",
    "PASSIVE_VIEWPOINT_OBSERVER_VERSION",
    "REAL_STAND_AXIS_OBSERVATION_KIND",
    "load_backside_axis_observation",
    "load_opposite_face_normal",
    "opposite_face_normal_from_axis_observation",
    "validated_backside_axis_observation",
]
