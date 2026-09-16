"""Reuse one canonical boundary within its exact current image, never a pose.

The transient proof retains pixel ownership and current raw-border evidence.
It is not serialized or retained by trackers. Recentring may translate that
same evidence into a view of the same image; another image must measure again.
"""

from dataclasses import replace
from hashlib import sha256
import math

from scripts.aufgabe04.perception.stand_axis.head_outer_border import validated_current_head_boundary
from scripts.aufgabe04.perception.stand_axis.models import ImagePoint


def _view_key(frame):
    return (tuple(frame.shape), frame.dtype.str, frame.strides,
            frame.__array_interface__["data"][0])


def _owner(frame):
    owner = frame
    while getattr(owner, "base", None) is not None:
        owner = owner.base
    return owner


def _same_view(first, second):
    return (_owner(first) is _owner(second) and _view_key(first) == _view_key(second))


def _digest(frame):
    return sha256(frame.tobytes()).digest()


def _shift(corners, dx, dy):
    return None if corners is None else tuple(ImagePoint(p.u_px - dx, p.v_px - dy) for p in corners)


class CurrentHeadRefinement:
    """In-memory complete current boundary, with no 3D or freshness authority."""

    def __init__(self, frame, raw_edges, *, model_profile, refinement, outer_recovery, seed):
        if (raw_edges.shape != frame.shape[:2] or not refinement.accepted
                or refinement.corners is None or refinement.support is None
                or refinement.corner_arm_support is None or not refinement.corner_arm_support.accepted
                or seed.corners != outer_recovery.neutral_proposal_corners
                or not validated_current_head_boundary(outer_recovery,
                    corners=refinement.corners, profile_sha256=model_profile.sha256)):
            raise ValueError("current head refinement requires complete measured boundary evidence")
        self._frame = frame
        self._frame_view = _view_key(frame)
        self._frame_digest = _digest(frame)
        self._raw_edges = raw_edges.copy()
        self._raw_edges.flags.writeable = False
        mask = refinement.evidence_mask.copy()
        mask.flags.writeable = False
        self._refinement = replace(refinement, evidence_mask=mask)
        self._outer_recovery = outer_recovery
        self._seed = seed
        self._model_profile = model_profile

    def _validate_context(self, frame, *, model_profile, proposal_corners):
        if (frame is None or not _same_view(self._frame, frame)
                or _view_key(frame) != self._frame_view
                or _digest(frame) != self._frame_digest
                or model_profile != self._model_profile
                or tuple(proposal_corners) != self._refinement.corners):
            raise ValueError("current head refinement belongs to another image, model or boundary")

    def raw_edges_for(self, frame, *, model_profile, proposal_corners):
        """Copy measured edges only for the exact unchanged image and boundary.

        Canny hysteresis can connect a selected border to strong pixels outside
        a recentered crop. Recomputing it on that crop can lose interior edges
        even though no source pixels changed. A rebased proof already retains
        the corresponding original measurement; fitting must consume it once.
        """
        self._validate_context(frame, model_profile=model_profile,
                               proposal_corners=proposal_corners)
        return self._raw_edges.copy()

    def resolve(self, frame, raw_edges, *, model_profile, proposal_corners):
        """Return the exact selected borders only while the original pixels persist."""
        import numpy as np

        self._validate_context(frame, model_profile=model_profile,
                               proposal_corners=proposal_corners)
        if raw_edges.shape != self._raw_edges.shape or raw_edges.dtype != self._raw_edges.dtype:
            raise ValueError("current head refinement belongs to another image, model or boundary")
        # Recentring changes Canny's image-padding behavior at the crop edge.
        # Check the complete selected head and a twelve-pixel support margin;
        # unchanged source pixels elsewhere cannot donate boundary evidence.
        corners = self._refinement.corners
        x0 = max(0, math.floor(min(p.u_px for p in corners)) - 12)
        y0 = max(0, math.floor(min(p.v_px for p in corners)) - 12)
        x1 = min(raw_edges.shape[1], math.ceil(max(p.u_px for p in corners)) + 13)
        y1 = min(raw_edges.shape[0], math.ceil(max(p.v_px for p in corners)) + 13)
        if not np.array_equal(raw_edges[y0:y1, x0:x1], self._raw_edges[y0:y1, x0:x1]):
            raise ValueError("current head refinement raw supporting pixels changed")
        # Debug consumers may draw on their own mask without mutating the proof.
        return (replace(self._refinement, evidence_mask=self._refinement.evidence_mask.copy()),
                self._outer_recovery, self._seed)

    def rebase(self, source_frame, target_frame, dx, dy):
        """Translate into an exact subview; dx/dy locate its origin in source pixels."""
        if (type(dx) is not int or type(dy) is not int or dx < 0 or dy < 0
                or not _same_view(self._frame, source_frame)
                or _view_key(source_frame) != self._frame_view
                or _digest(source_frame) != self._frame_digest):
            raise ValueError("current head refinement crop source changed")
        height, width = target_frame.shape[:2]
        if (dy + height > source_frame.shape[0] or dx + width > source_frame.shape[1]
                or not _same_view(source_frame[dy:dy+height, dx:dx+width], target_frame)):
            raise ValueError("current head refinement needs an exact source-image subview")
        corners = _shift(self._refinement.corners, dx, dy)
        if not all(0 <= p.u_px < width and 0 <= p.v_px < height for p in corners):
            raise ValueError("current head refinement crop clips the selected head")
        measurement = replace(self._refinement, corners=corners,
            candidate_corners=_shift(self._refinement.candidate_corners, dx, dy),
            evidence_mask=self._refinement.evidence_mask[dy:dy+height, dx:dx+width])
        outer = replace(self._outer_recovery,
            original_corners=_shift(self._outer_recovery.original_corners, dx, dy),
            recovered_corners=_shift(self._outer_recovery.recovered_corners, dx, dy),
            neutral_proposal_corners=_shift(self._outer_recovery.neutral_proposal_corners, dx, dy),
            current_raw_alternatives=tuple(_shift(quad, dx, dy)
                for quad in self._outer_recovery.current_raw_alternatives))
        return CurrentHeadRefinement(target_frame,
            self._raw_edges[dy:dy+height, dx:dx+width], model_profile=self._model_profile,
            refinement=measurement, outer_recovery=outer,
            seed=replace(self._seed, corners=_shift(self._seed.corners, dx, dy)))


def capture_current_head_refinement(frame_bgr, raw_edges, *, model_profile,
                                    refinement, outer_recovery, seed):
    """Bind accepted canonical raw evidence to this image before pose fitting."""
    return CurrentHeadRefinement(frame_bgr, raw_edges, model_profile=model_profile,
        refinement=refinement, outer_recovery=outer_recovery, seed=seed)
