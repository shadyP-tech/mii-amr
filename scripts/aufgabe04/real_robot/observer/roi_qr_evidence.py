"""Preserve conservative QR evidence when current-image processing changes ROI.

Only the selected crop may supply identity to the existing target binding.
This summary adds veto/conflict evidence: recentering must not erase another
current crop's decoded symbol or verified front marker. No decoder or fit runs
here, and no text is promoted to candidate identity.
"""

from dataclasses import asdict, dataclass

from scripts.aufgabe04.qr_scanning.qr_observation import validated_qr_corners


@dataclass(frozen=True)
class RoiQrEvidence:
    marker_detected: bool
    marker_verified: bool | None
    qr_texts: tuple[str, ...]
    conflict_reason: str | None
    symbol_count: int

    def metadata(self) -> dict:
        return {**asdict(self), "motion_authorized": False,
                "identity_authority": "selected_crop_target_binding_only"}


def _signed_area(polygon):
    return sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(polygon, (*polygon[1:], polygon[0]))) / 2.0


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersection_area(subject, clip):
    """Clip two validated convex quads without an OpenCV dependency."""
    output = list(subject)
    orientation = 1.0 if _signed_area(clip) > 0 else -1.0
    for edge_start, edge_end in zip(clip, (*clip[1:], clip[0])):
        if not output:
            return 0.0
        previous, values = output[-1], output
        output = []
        previous_side = orientation * _cross(edge_start, edge_end, previous)
        for current in values:
            current_side = orientation * _cross(edge_start, edge_end, current)
            if (current_side >= 0) != (previous_side >= 0):
                fraction = previous_side / (previous_side - current_side)
                output.append((previous[0] + fraction * (current[0] - previous[0]),
                               previous[1] + fraction * (current[1] - previous[1])))
            if current_side >= 0:
                output.append(current)
            previous, previous_side = current, current_side
    return abs(_signed_area(output)) if len(output) >= 3 else 0.0


def _same_symbol(first, second):
    # Different decoders/crops can move QR corners slightly. Mere bounding-box
    # overlap or touching edges cannot collapse two distinct printed symbols.
    smaller = min(abs(_signed_area(first)), abs(_signed_area(second)))
    return smaller > 0 and _intersection_area(first, second) / smaller >= 0.5


def _full_image_corners(evaluation, observation):
    roi = evaluation.attempt.roi
    local = validated_qr_corners(
        observation.corners,
        image_shape=(roi.y1 - roi.y0, roi.x1 - roi.x0),
    )
    if local is None:
        return None
    return tuple((u + roi.x0, v + roi.y0) for u, v in local)


def summarize_roi_qr_evidence(registration) -> RoiQrEvidence:
    """Summarize every evaluated crop of one current image for vetoes only.

    ``None`` verification is retained only for legacy marker producers that
    supply no explicit verification result. Operational producers always use
    booleans. Text without localized corners still verifies marker presence;
    its candidate identity remains unbound unless the selected crop's ordinary
    binding independently succeeds.
    """
    evaluations = registration.evaluations
    texts: set[str] = set()
    marker_detected = verified = legacy_marker = False
    within_roi_count = 0
    groups: list[list[tuple[tuple[float, float], ...]]] = []
    for evaluation in evaluations:
        observations = evaluation.qr_observations or ()
        detected = evaluation.debug.qr_detected is True
        verification = getattr(evaluation.debug, "qr_marker_verified", None)
        marker_detected |= detected or bool(observations) or verification is True
        verified |= bool(observations) or verification is True
        legacy_marker |= detected and verification is None
        within_roi_count = max(within_roi_count, len(observations))
        for observation in observations:
            texts.add(observation.text)
            corners = _full_image_corners(evaluation, observation)
            if corners is None:
                continue
            # Complete agreement avoids a chain of overlapping noisy boxes
            # merging disjoint first/last symbols across three evaluations.
            for group in groups:
                if all(_same_symbol(corners, other) for other in group):
                    group.append(corners)
                    break
            else:
                groups.append([corners])
    if within_roi_count > 1:
        conflict = "multiple_qr_symbols_in_evaluated_roi"
    elif len(texts) > 1:
        conflict = "conflicting_qr_identities_across_rois"
    elif len(groups) > 1:
        conflict = "disjoint_qr_symbols_across_rois"
    else:
        conflict = None
    return RoiQrEvidence(
        marker_detected=marker_detected,
        marker_verified=True if verified else (None if legacy_marker else False),
        qr_texts=tuple(sorted(texts)), conflict_reason=conflict,
        symbol_count=max(within_roi_count, len(texts), len(groups)),
    )
